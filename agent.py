from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    google,
    openai,
    cartesia,
    silero,
    noise_cancellation,# noqa: F401
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel



# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("invoice-validation-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")


class InvoiceValidationAgent(Agent):
    def __init__(
        self,
        *,
        dealership_name: str,
        invoice_number: str,
        customer_name: str,
        vin: str,
        needs_email: bool,
        metadata: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            YOU ONLY SPEAKS IN SPANISH, You are Alicia, a professional and courteous representative from Nexcar Financiera calling in Spanish.
            You are calling {dealership_name} to verify invoice information. 

            Invoice Details:
            - Invoice Number: {invoice_number}
            - Customer Name: {customer_name}
            - VIN: {vin}

            Your primary goal: {"Collect the dealership's email address to send the validation request." if needs_email else "Verify the invoice details with the dealership."}

            Be polite, professional, and brief. If they need to transfer you to someone else, politely wait.
            If you successfully collect the email or verify the information, use the collect_email tool to save it.
            Allow the dealership representative to end the conversation naturally, any number or serial number tell it slowly so the people can understand you.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.metadata = metadata
        self.needs_email = needs_email

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def collect_email(
        self,
        ctx: RunContext,
        email: str,
    ):
        """Called when the dealership provides their email address for invoice validation.

        Args:
            email: The email address provided by the dealership
        """
        logger.info(f"Email collected from {self.participant.identity}: {email}")

        # Store the collected email in metadata for later processing
        validation_id = self.metadata.get("validationId")
        dealership_id = self.metadata.get("dealershipId")

        logger.info(f"Collected email for validation {validation_id}, dealership {dealership_id}")

        # Call the backend to save the email
        import os
        import json
        try:
            base_url = os.getenv("BACKEND_URL", "http://localhost:3000")
            import urllib.request

            data = json.dumps({
                "validationId": validation_id,
                "email": email,
                "additionalData": {}
            }).encode('utf-8')

            req = urllib.request.Request(
                f"{base_url}/api/validation-info-collected",
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req) as response:
                logger.info(f"Email saved successfully: {response.status}")

        except Exception as e:
            logger.error(f"Error saving email to backend: {e}")

        return f"Email {email} has been recorded. Thank you!"

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        logger.info(f"ending the call for {self.participant.identity}")

        # let the agent finish speaking
        current_speech = ctx.session.current_speech
        if current_speech:
            await current_speech.wait_for_playout()

        await self.hangup()

    @function_tool()
    async def confirm_invoice_details(
        self,
        ctx: RunContext,
        confirmed: bool,
        notes: str = "",
    ):
        """Called when the dealership confirms or denies the invoice details.

        Args:
            confirmed: True if dealership confirms the invoice is valid, False otherwise
            notes: Any additional notes or comments from the dealership
        """
        logger.info(
            f"Invoice confirmation from {self.participant.identity}: {confirmed}, notes: {notes}"
        )

        validation_id = self.metadata.get("validationId")
        logger.info(f"Invoice details confirmed for validation {validation_id}")

        return f"Thank you for {'confirming' if confirmed else 'providing feedback on'} the invoice details."

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()


async def entrypoint(ctx: JobContext):
    logger.info(f"Starting agent for room {ctx.room.name}")

    # Parse the metadata sent from the frontend
    # Metadata includes: validationId, dealershipId, requestId, invoiceData, phoneNumber, needsEmail
    metadata = json.loads(ctx.job.metadata)
    logger.info(f"Received metadata: {metadata}")

    phone_number = metadata.get("phoneNumber")
    participant_identity = f"dealership-{metadata.get('dealershipId', 'unknown')}"

    # Extract invoice data
    invoice_data = metadata.get("invoiceData", {})
    dealership_name = invoice_data.get("dealershipName", "Unknown Dealership")
    invoice_number = invoice_data.get("invoiceNumber", "N/A")
    customer_name = invoice_data.get("customerName", "Unknown Customer")
    vin = invoice_data.get("vin", "N/A")
    needs_email = metadata.get("needsEmail", True)

    # Step 1: Connect to the room (as per LiveKit docs)
    logger.info("Connecting to room...")
    await ctx.connect()

    # Step 2: Place outbound call (if phone number provided)
    if phone_number is not None:
        logger.info(f"Placing outbound call to {phone_number}...")
        try:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=ctx.room.name,
                    sip_trunk_id=outbound_trunk_id,
                    sip_call_to=phone_number,
                    participant_identity=participant_identity,
                    # function blocks until user answers the call, or if the call fails
                    wait_until_answered=True,
                )
            )
            logger.info("Call answered successfully")
        except api.TwirpError as e:
            logger.error(
                f"error creating SIP participant: {e.message}, "
                f"SIP status: {e.metadata.get('sip_status_code')} "
                f"{e.metadata.get('sip_status')}"
            )
            ctx.shutdown()
            return

    # Step 3: Create the invoice validation agent with actual metadata
    agent = InvoiceValidationAgent(
        dealership_name=dealership_name,
        invoice_number=invoice_number,
        customer_name=customer_name,
        vin=vin,
        needs_email=needs_email,
        metadata=metadata,
    )

    # Step 4: Start agent session (after SIP participant is connected)
    session = AgentSession(
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        tts=cartesia.TTS(model="sonic-2", voice="5c5ad5e7-1020-476b-8b91-fdcbe9cc313c"),
        llm=google.LLM(model="gemini-2.0-flash-exp",),
    )

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # enable Krisp background voice and noise removal
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Wait for participant to fully join
    participant = await ctx.wait_for_participant(identity=participant_identity)
    logger.info(f"participant joined: {participant.identity}")
    agent.set_participant(participant)

    # For outbound calls, wait for recipient to speak first
    # Agent will automatically respond after recipient's turn ends
    # (Do NOT call generate_reply for outbound calls)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="invoice-validation-agent",
        )
    )