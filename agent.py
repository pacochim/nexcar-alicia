from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any
from datetime import datetime

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
from supabase import create_client, Client



# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("invoice-validation-agent")
logger.setLevel(logging.INFO)

outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None


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
            IMPORTANTE: HABLAS ÚNICAMENTE EN ESPAÑOL. Eres Alicia, representante profesional y cortés de Nexcar Financiera.

            CONTEXTO DE LA LLAMADA:
            Estás llamando a {dealership_name} porque un cliente está solicitando un crédito automotriz con Nexcar Financiera.
            El cliente nos presentó una factura que supuestamente fue emitida por {dealership_name}, y necesitas verificar que sea auténtica.

            INFORMACIÓN DE LA FACTURA QUE SUPUESTAMENTE ELLOS EMITIERON:
            - Número de Factura: {invoice_number}
            - Nombre del Cliente: {customer_name}
            - VIN (Número de Serie del Vehículo): {vin}

            TU OBJETIVO PRINCIPAL:
            {"Solicitar un correo electrónico para enviarles la factura que el cliente nos proporcionó y que supuestamente ellos emitieron. Explica que necesitas que ellos la revisen y confirmen si realmente la emitieron o no, como parte de la validación del crédito." if needs_email else "Verificar los detalles de la factura con el concesionario."}

            GUION SUGERIDO PARA SOLICITAR EMAIL:
            'Hola, le llamo de Nexcar Financiera. Tenemos un cliente solicitando un crédito y nos presentó una factura que dice ser de su concesionario.
            Para verificar que sea auténtica, ¿me podría proporcionar un correo electrónico para enviarles la factura y que ustedes la revisen?
            Necesitamos confirmar que realmente la emitieron.'

            INSTRUCCIONES CRÍTICAS PARA NÚMEROS Y DELETREO:
            - SIEMPRE deletrea números de serie (VIN, números de factura) LETRA POR LETRA en español
            - Para números sueltos (1, 2, 3...), di "uno", "dos", "tres" NUNCA "one", "two", "three"
            - Di los números LENTAMENTE con PAUSAS entre cada carácter
            - Ejemplo VIN: "el vin es: A... B... C... uno... dos... tres... cuatro... D... E... F"
            - Ejemplo factura: "número de factura: F... A... C... T... guión... dos... cero... dos... cinco"
            - Al deletrear emails, usa: "arroba" para @, "punto" para ., "guión bajo" para _
            - Ejemplo email: "ventas... arroba... ejemplo... punto... com"

            MODISMOS Y EXPRESIONES EN ESPAÑOL MEXICANO:
            - Usa "¿Cómo está?" o "¿Cómo le va?" en lugar de "¿Cómo estás?"
            - Di "Con mucho gusto" en lugar de "De nada"
            - Usa "Claro que sí" o "Por supuesto" para confirmaciones
            - Di "Disculpe" en lugar de "Perdón" o "Sorry"
            - Usa "En un momento" en lugar de "Espere"
            - Di "¿Me permite?" antes de pedir información
            - Termina con "Que tenga buen día" o "Hasta luego, que esté bien"

            MANEJO DE INTERRUPCIONES Y ACLARACIONES:
            - Si no entiendes algo, di: "Disculpe, ¿me podría repetir eso por favor?"
            - Si necesitan tiempo, di: "Sin problema, tómese su tiempo"
            - Si te transfieren, di: "Perfecto, quedo en espera. Muchas gracias"
            - Para confirmar info, repite: "Entonces, ¿el correo es...?" (repite despacio)
            - Si hay ruido/interferencia: "Disculpe, se escucha un poco cortado, ¿me podría repetir?"

            VOCABULARIO TÉCNICO EN ESPAÑOL:
            - VIN = "vin" o "número de serie del vehículo" (nunca "vehicle identification number")
            - Invoice = "factura" (nunca "invoice")
            - Email = "correo electrónico" o "correo" (nunca "email" en inglés)
            - Dealership = "concesionario" o "agencia" (nunca "dealership")
            - Customer = "cliente" (nunca "customer")

            PAUSAS Y RITMO DE CONVERSACIÓN:
            - Haz pausas breves después de preguntar algo (1-2 segundos)
            - Cuando deletrees, haz PAUSAS claras entre cada letra/número
            - Si mencionas varios datos, sepáralos con pausas: "El número de factura es... [pausa] F-A-C-T..."
            - Después de dar información compleja, pregunta: "¿Pudo anotarlo?"
            - Si están anotando, di: "Le voy despacio..." y reduce velocidad
            - Permite interrupciones naturales - no hables en párrafos largos

            CONFIRMACIONES Y REPETICIONES:
            - Siempre confirma el correo deletreándolo de vuelta: "Perfecto, entonces el correo es: ventas... arroba... toyota... punto... com, ¿es correcto?"
            - Si algo suena dudoso, confirma: "¿Me dijo 'toyota' o 'toyoda'?"
            - Usa frases de confirmación: "Entendido", "Perfecto", "Muy bien"
            - Para asegurar: "¿Me lo puede repetir para confirmar?"

            MANEJO DE SITUACIONES COMUNES:
            - Si dicen "no tengo el email aquí": "Sin problema, ¿me podría comunicar con alguien que lo tenga?"
            - Si dicen "llame más tarde": "Claro que sí, ¿a qué hora me recomienda llamar?"
            - Si dicen "envíeme un mensaje": "Con gusto, pero necesito primero el correo para enviarle la factura"
            - Si preguntan "¿quién es?": Repite claramente tu nombre y empresa
            - Si están ocupados: "Entiendo que está ocupado, ¿sería mejor llamar en otro momento?"
            - Si dicen que no reconocen la factura: "Justamente por eso llamamos, para verificarlo con ustedes"

            TONO Y COMPORTAMIENTO:
            - Sé cordial, profesional y breve
            - Habla de manera natural, como una persona mexicana profesional
            - Evita sonar robótica - usa muletillas naturales como "este...", "bueno...", "entonces..."
            - Si necesitan transferirte con otra persona, espera pacientemente
            - Cuando recopiles el email exitosamente, usa la herramienta collect_email para guardarlo
            - Permite que el representante termine la conversación naturalmente
            - Mantén un tono de colaboración - están ayudando en el proceso de verificación
            - Deja claro que necesitas que ELLOS CONFIRMEN si la factura es real o no
            - NO uses palabras en inglés bajo ninguna circunstancia
            - Adapta tu velocidad al ritmo de la conversación - más lento si están anotando
            - Sé paciente y amable incluso si están apurados o confundidos
            - Sonríe al hablar - se nota en la voz (mantén tono positivo)

            CIERRE DE LLAMADA:
            - Agradece siempre: "Le agradezco mucho su tiempo y su ayuda"
            - Confirma próximos pasos: "Le enviaremos la factura al correo que me proporcionó"
            - Despedida cordial: "Que tenga excelente día" o "Hasta luego, quedo al pendiente"
            - NO cuelgues abruptamente - espera a que ellos se despidan también
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
        """Utiliza esta herramienta cuando el concesionario proporcione su dirección de correo electrónico para enviar la factura que necesitamos verificar.

        Args:
            email: La dirección de correo electrónico proporcionada por el concesionario
        """
        logger.info(f"Email collected from {self.participant.identity}: {email}")

        # Store the collected email in metadata for later processing
        # This will be saved to the database at the end of the call
        self.metadata["collectedEmail"] = email

        return f"El correo electrónico {email} ha sido registrado exitosamente. ¡Muchas gracias por su colaboración!"

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Utiliza esta herramienta cuando el representante del concesionario indique que desea terminar la llamada o cuando ya hayas completado tu objetivo"""
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


async def save_call_transcript(session: AgentSession, metadata: dict, room_name: str):
    """Save the call transcript to the database after call ends and trigger next steps"""
    try:
        if not supabase:
            logger.error("Supabase client not initialized")
            return

        # Get the full transcript from session.chat_ctx (as per LiveKit docs)
        transcript_parts = []

        if hasattr(session, 'chat_ctx') and session.chat_ctx:
            # Extract messages from chat context
            for message in session.chat_ctx.messages:
                role = message.role

                # Skip system messages
                if role == 'system':
                    continue

                # Get content - handle both string and list of content items
                content = ""
                if isinstance(message.content, str):
                    content = message.content
                elif isinstance(message.content, list):
                    # Join all text content items
                    content = " ".join([
                        item.text if hasattr(item, 'text') else str(item)
                        for item in message.content
                    ])

                # Format role name
                role_name = "Agent" if role == "assistant" else "Dealership"
                transcript_parts.append(f"{role_name}: {content}")

        full_transcript = "\n".join(transcript_parts)

        logger.info(f"Saving transcript for room {room_name}: {len(full_transcript)} characters")

        validation_id = metadata.get("validationId")
        dealership_id = metadata.get("dealershipId")
        collected_email = metadata.get("collectedEmail")

        # Calculate call duration if we have start time
        call_ended_at = datetime.utcnow().isoformat()

        # Insert call record into database
        call_data = {
            "validation_id": validation_id,
            "dealership_id": dealership_id,
            "room_name": room_name,
            "full_transcript": full_transcript,
            "call_ended_at": call_ended_at,
            "email_collected": collected_email,
            "call_outcome": "success" if collected_email else "failed",
        }

        result = supabase.table("calls").insert(call_data).execute()

        if result.data:
            logger.info(f"✅ Call transcript saved successfully: {result.data[0]['id']}")

            # Update validation status to ENDED_CALL for later processing
            update_result = supabase.table("validations").update({
                "status": "ENDED_CALL",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", validation_id).execute()

            if update_result.data:
                logger.info(f"✅ Validation status updated to ENDED_CALL for validation: {validation_id}")
            else:
                logger.error(f"❌ Failed to update validation status for: {validation_id}")
        else:
            logger.error("Failed to save call transcript")

    except Exception as e:
        logger.error(f"❌ Error saving call transcript: {e}")


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

    # Register callback to save transcript when session ends (as per LiveKit docs)
    async def write_transcript():
        logger.info("Call ended, saving transcript...")
        await save_call_transcript(session, metadata, ctx.room.name)

    ctx.add_shutdown_callback(write_transcript)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="invoice-validation-agent",
        )
    )