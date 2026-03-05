from __future__ import annotations

import asyncio
import logging
import weakref
import time
from dataclasses import dataclass, replace
from typing import AsyncGenerator, Optional

import aiohttp

from livekit import rtc
from livekit.agents import tokenize, tts, utils
from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.agents.types import (
    APIConnectOptions,
    DEFAULT_API_CONNECT_OPTIONS,
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NUM_CHANNELS = 1
DEFAULT_SAMPLE_RATE = 24000  # Match your FastAPI app's SAMPLE_RATE
DEFAULT_BASE_URL = "http://localhost:8080"  # Your FastAPI app port


@dataclass
class _TTSOptions:
    """Internal dataclass to hold TTS options."""
    base_url: str
    voice: str
    language: str
    temperature: float
    max_tokens: int
    top_p: float
    repetition_penalty: float
    sample_rate: int
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    """
    A TTS implementation for the FastAPI PCM streaming endpoint.
    This class communicates directly with your FastAPI TTS server.
    """
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        voice: str = "tara",
        language: str = "en",
        temperature: float = 0.6,
        max_tokens: int = 1800,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        word_tokenizer: tokenize.WordTokenizer | None = None,
        connect_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Initialize the FastAPI TTS engine.

        Args:
            base_url (str): Base URL for your FastAPI TTS server.
            voice (str): Voice identifier (default: "tara").
            language (str): Language code (default: "en").
            temperature (float): Sampling temperature for text generation.
            max_tokens (int): Maximum tokens to generate.
            top_p (float): Top-p sampling parameter.
            repetition_penalty (float): Repetition penalty for text generation.
            sample_rate (int): Sample rate in Hz.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text.
            connect_options (APIConnectOptions): Connection options for the API.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        # Use a basic word tokenizer if none is provided
        if not word_tokenizer:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        self._opts = _TTSOptions(
            base_url=base_url.rstrip('/'),
            voice=voice,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            sample_rate=sample_rate,
            word_tokenizer=word_tokenizer,
        )

        self._session: aiohttp.ClientSession | None = None
        self._connect_options = connect_options
        self._streams = weakref.WeakSet["SynthesizeStream"]()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> "ChunkedStream":
        """Synthesizes text into audio using the streaming PCM endpoint."""
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or self._connect_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> "SynthesizeStream":
        """Creates a streaming synthesis task."""
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options or self._connect_options
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Pre-warm connection to the TTS service"""
        logger.info(f"Pre-warming TTS connection to {self._opts.base_url}")
        # Create a session early to establish connection pool
        if self._session is None:
            asyncio.create_task(self._ensure_session())

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensures an active aiohttp session is available."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(
                limit=10,  # Connection pool limit
                limit_per_host=5,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
        return self._session

    async def _health_check(self) -> bool:
        """Check if the TTS service is healthy"""
        try:
            session = await self._ensure_session()
            url = f"{self._opts.base_url}/health"
            async with session.get(url) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    async def aclose(self) -> None:
        """Closes the TTS instance and cleans up resources."""
        # Close all active streams
        streams_to_close = list(self._streams)
        if streams_to_close:
            logger.info(f"Closing {len(streams_to_close)} active streams")
            await asyncio.gather(
                *[stream.aclose() for stream in streams_to_close],
                return_exceptions=True
            )

        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("TTS session closed")


class ChunkedStream(tts.ChunkedStream):
    """A chunked stream for non-streaming synthesis."""
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(self._tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        The main task for the chunked stream.
        This method fetches PCM audio directly from the FastAPI endpoint.
        """
        request_id = utils.shortuuid()
        
        # Initialize the output emitter
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )
        
        try:
            bytes_received = 0
            async for audio_bytes in self._fetch_pcm_audio(self._input_text):
                # Push raw PCM audio data directly to the emitter
                output_emitter.push(audio_bytes)
                bytes_received += len(audio_bytes)
                
            logger.debug(f"Chunked stream complete: {bytes_received} bytes received for request {request_id}")
            
        except Exception as e:
            logger.error(f"TTS API error in chunked stream {request_id}: {e}")
            raise tts.APIError(f"Error during chunked TTS synthesis: {e}") from e

    async def _fetch_pcm_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Fetches raw PCM audio from the FastAPI /v1/audio/speech endpoint."""
        session = await self._tts._ensure_session()
        request_data = {
            "prompt": text,
            "voice": self._opts.voice,
            "language": self._opts.language,
            "temperature": self._opts.temperature,
            "max_tokens": self._opts.max_tokens,
            "top_p": self._opts.top_p,
            "repetition_penalty": self._opts.repetition_penalty,
        }
        url = f"{self._opts.base_url}/v1/audio/speech"

        logger.info(f"Requesting TTS for text: {text[:50]}...")

        try:
            async with session.post(
                url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise tts.APIError(f"TTS API returned {response.status}: {error_text}")
                
                # Stream raw PCM bytes directly
                total_bytes = 0
                async for chunk in response.content.iter_chunked(4096):  # 4KB chunks
                    if chunk:
                        total_bytes += len(chunk)
                        yield chunk
                
                logger.debug(f"Received {total_bytes} bytes of PCM audio")
                        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise APIConnectionError(f"HTTP client error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in _fetch_pcm_audio: {e}")
            raise tts.APIError(f"Unexpected error: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """A stream for synthesizing audio in real-time."""
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(self._tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    def _mark_started(self) -> None:
        """Override parent method to ensure proper metrics collection timing"""
        super()._mark_started()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        The main task for the synthesis stream.
        This method processes input text segments and streams PCM audio.
        """
        request_id = utils.shortuuid()
        
        # Initialize the output emitter for streaming
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=True,
        )

        # Start the input tokenization task
        input_task = asyncio.create_task(self._tokenize_input())
        segment_id_counter = 0
        
        try:
            # Process each segment of text from the tokenizer
            async for word_stream in self._segments_ch:
                # Collect tokens and reconstruct text with proper spacing
                tokens = []
                async for ev in word_stream:
                    tokens.append(ev.token)
                
                # Use the word tokenizer's format_words method to properly reconstruct text
                segment_text = self._opts.word_tokenizer.format_words(tokens)
                
                if not segment_text.strip():
                    continue
                
                # Mark started for metrics collection timing
                self._mark_started()
                
                segment_id = f"{request_id}-{segment_id_counter}"
                segment_id_counter += 1
                
                logger.info(f"Processing segment {segment_id}: {segment_text[:50]}...")
                
                # Start a segment for this text
                output_emitter.start_segment(segment_id=segment_id)

                bytes_received = 0
                async for audio_bytes in self._fetch_pcm_audio(segment_text):
                    # Push raw PCM audio data directly to the emitter
                    output_emitter.push(audio_bytes)
                    bytes_received += len(audio_bytes)
                
                # Mark the end of this segment
                output_emitter.end_segment()
                
                logger.debug(f"Segment {segment_id} complete: {bytes_received} bytes")
                
        except Exception as e:
            logger.error(f"TTS API error in synthesis stream: {e}")
            raise tts.APIError(f"Error during streaming TTS synthesis: {e}") from e
        finally:
            await utils.aio.gracefully_cancel(input_task)

    @utils.log_exceptions(logger=logger)
    async def _tokenize_input(self):
        """
        Reads from the input channel, tokenizes text into words/sentences,
        and pushes word streams to the segments channel for processing.
        """
        word_stream = None
        async for data in self._input_ch:
            if isinstance(data, str):
                if word_stream is None:
                    word_stream = self._opts.word_tokenizer.stream()
                    self._segments_ch.send_nowait(word_stream)
                word_stream.push_text(data)
            elif isinstance(data, self._FlushSentinel):
                if word_stream:
                    word_stream.end_input()
                word_stream = None
        
        if word_stream:
            word_stream.end_input()

        self._segments_ch.close()
    
    async def _fetch_pcm_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Fetches raw PCM audio from the FastAPI /v1/audio/speech endpoint."""
        session = await self._tts._ensure_session()
        request_data = {
            "prompt": text,
            "voice": self._opts.voice,
            "language": self._opts.language,
            "temperature": self._opts.temperature,
            "max_tokens": self._opts.max_tokens,
            "top_p": self._opts.top_p,
            "repetition_penalty": self._opts.repetition_penalty,
        }
        url = f"{self._opts.base_url}/v1/audio/speech"

        try:
            async with session.post(
                url,
                json=request_data,
                headers={'Content-Type': 'application/json'},
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise tts.APIError(f"TTS API returned {response.status}: {error_text}")
                
                # Stream raw PCM bytes directly
                total_bytes = 0
                async for chunk in response.content.iter_chunked(4096):  # 4KB chunks
                    if chunk:
                        total_bytes += len(chunk)
                        yield chunk
                
                logger.debug(f"Received {total_bytes} bytes of PCM audio for streaming")
                        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise APIConnectionError(f"HTTP client error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in streaming _fetch_pcm_audio: {e}")
            raise tts.APIError(f"Unexpected error: {e}") from e
            #raise tts.TTSError(f"Unexpected error: {e}") from e

    async def aclose(self) -> None:
        """Close the stream and clean up resources"""
        logger.debug("Closing SynthesizeStream")
        await super().aclose()
        self._segments_ch.close()


# Alias for easier import
FastAPITTS = TTS