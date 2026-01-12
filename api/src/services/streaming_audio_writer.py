"""Audio conversion service with proper streaming support and Base64 encoding"""

import base64
import struct
from io import BytesIO
from typing import Optional, Union, Literal

import av
import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment


class StreamingAudioWriter:
    """Handles streaming audio format conversions with optional Base64 encoding"""

    def __init__(
        self,
        format: str,
        sample_rate: int,
        channels: int = 1,
        base64_encode: bool = False,
        base64_output_type: Literal["string", "bytes"] = "string",
        include_data_uri: bool = False,
    ):
        """
        Initialize the streaming audio writer.

        Args:
            format: Audio format ('wav', 'mp3', 'opus', 'flac', 'aac', 'pcm')
            sample_rate: Sample rate in Hz (e.g., 24000, 44100)
            channels: Number of audio channels (1 for mono, 2 for stereo)
            base64_encode: If True, return Base64-encoded output instead of raw bytes
            base64_output_type: Return Base64 as 'string' or 'bytes'
            include_data_uri: If True and base64_encode is True, prefix with data URI
                              (e.g., 'data:audio/mp3;base64,')
        """
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0

        # Base64 encoding options
        self.base64_encode = base64_encode
        self.base64_output_type = base64_output_type
        self.include_data_uri = include_data_uri

        # MIME type mapping for data URIs
        self._mime_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "aac": "audio/aac",
            "pcm": "audio/pcm",
        }

        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "mp3",
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }

        # Format-specific setup
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format != "pcm":
                self.output_buffer = BytesIO()
                container_options = {}

                # Try disabling Xing VBR header for MP3 to fix iOS timeline reading issues
                if self.format == "mp3":
                    container_options = {"write_xing": "0"}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")

                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options,
                )
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )

                # Set bit_rate only for codecs where it's applicable
                if self.format in ["mp3", "aac", "opus"]:
                    self.stream.bit_rate = 160000
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _encode_base64(self, data: bytes) -> Union[str, bytes]:
        """
        Encode bytes to Base64.

        Args:
            data: Raw bytes to encode

        Returns:
            Base64-encoded string or bytes depending on base64_output_type
        """
        if not data:
            return "" if self.base64_output_type == "string" else b""

        encoded = base64.b64encode(data)

        if self.base64_output_type == "string":
            encoded_str = encoded.decode("ascii")
            if self.include_data_uri:
                mime_type = self._mime_types.get(self.format, "application/octet-stream")
                return f"data:{mime_type};base64,{encoded_str}"
            return encoded_str
        else:
            if self.include_data_uri:
                mime_type = self._mime_types.get(self.format, "application/octet-stream")
                prefix = f"data:{mime_type};base64,".encode("ascii")
                return prefix + encoded
            return encoded

    def get_mime_type(self) -> str:
        """Get the MIME type for the current audio format."""
        return self._mime_types.get(self.format, "application/octet-stream")

    def close(self):
        """Close the container and output buffer."""
        if hasattr(self, "container"):
            self.container.close()

        if hasattr(self, "output_buffer"):
            self.output_buffer.close()

    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> Union[bytes, str]:
        """
        Write a chunk of audio data and return bytes/Base64 in the target format.

        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream

        Returns:
            Raw bytes or Base64-encoded string/bytes depending on configuration
        """
        if finalize:
            if self.format != "pcm":
                # Defensive check: if buffer is already closed, return empty
                if not hasattr(self, "output_buffer") or self.output_buffer.closed:
                    logger.warning(
                        "Buffer already closed during finalization, returning empty data"
                    )
                    return self._encode_base64(b"") if self.base64_encode else b""

                # Get the buffer data FIRST, before any operations that might close it
                data = self.output_buffer.getvalue()

                try:
                    # Flush the encoder by encoding None
                    packets = self.stream.encode(None)
                    for packet in packets:
                        self.container.mux(packet)

                    logger.debug("Muxed final packets.")

                    # If flush succeeded AND buffer is still open, get any additional data
                    if not self.output_buffer.closed:
                        additional_data = self.output_buffer.getvalue()[len(data) :]
                        if additional_data:
                            data += additional_data
                    else:
                        logger.debug(
                            "Buffer closed after muxing, using data captured before flush"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error during final encode flush (may be harmless): {e}"
                    )

                # Close the container - this writes the trailer
                try:
                    self.container.close()
                except Exception as e:
                    logger.warning(f"Error closing container (may be harmless): {e}")

                # Close the buffer
                if hasattr(self, "output_buffer") and not self.output_buffer.closed:
                    self.output_buffer.close()

                return self._encode_base64(data) if self.base64_encode else data
            else:
                return self._encode_base64(b"") if self.base64_encode else b""

        if audio_data is None or len(audio_data) == 0:
            return self._encode_base64(b"") if self.base64_encode else b""

        if self.format == "pcm":
            # Write raw bytes
            data = audio_data.tobytes()
            return self._encode_base64(data) if self.base64_encode else data
        else:
            # Defensive check: if buffer is closed, we can't write more data
            if not hasattr(self, "output_buffer") or self.output_buffer.closed:
                logger.error("Buffer is closed, cannot write audio data")
                return self._encode_base64(b"") if self.base64_encode else b""

            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo",
            )
            frame.sample_rate = self.sample_rate

            frame.pts = self.pts
            self.pts += frame.samples

            packets = self.stream.encode(frame)
            for packet in packets:
                self.container.mux(packet)

            # Check if buffer is still open after muxing
            if self.output_buffer.closed:
                logger.error("Buffer was closed after muxing packet")
                return self._encode_base64(b"") if self.base64_encode else b""

            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)

            return self._encode_base64(data) if self.base64_encode else data


# Convenience function for one-shot Base64 encoding of audio
def audio_to_base64(
    audio_data: np.ndarray,
    format: str = "mp3",
    sample_rate: int = 24000,
    channels: int = 1,
    include_data_uri: bool = False,
) -> str:
    """
    Convert audio data to Base64-encoded string in one shot.

    Args:
        audio_data: NumPy array of audio samples (int16)
        format: Output format ('mp3', 'wav', 'opus', 'flac', 'aac')
        sample_rate: Sample rate in Hz
        channels: Number of channels
        include_data_uri: If True, prefix with data URI scheme

    Returns:
        Base64-encoded audio string
    """
    writer = StreamingAudioWriter(
        format=format,
        sample_rate=sample_rate,
        channels=channels,
        base64_encode=True,
        base64_output_type="string",
        include_data_uri=include_data_uri,
    )

    # Write all audio data
    result = writer.write_chunk(audio_data)

    # Finalize and append any remaining data
    final_chunk = writer.write_chunk(finalize=True)

    # Combine results (handling data URI prefix properly)
    if include_data_uri and result and final_chunk:
        # Remove data URI prefix from final chunk if present
        if final_chunk.startswith("data:"):
            final_chunk = final_chunk.split(",", 1)[1] if "," in final_chunk else ""
        return result + final_chunk
    elif result and final_chunk:
        return result + final_chunk
    elif result:
        return result
    else:
        return final_chunk


# Example usage
if __name__ == "__main__":
    import numpy as np

    # Generate a simple test tone (440 Hz sine wave)
    sample_rate = 24000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Example 1: Streaming with Base64 output
    print("=== Streaming Base64 Example ===")
    writer = StreamingAudioWriter(
        format="mp3",
        sample_rate=sample_rate,
        channels=1,
        base64_encode=True,
        base64_output_type="string",
        include_data_uri=False,
    )

    # Write in chunks
    chunk_size = 4800  # 200ms chunks at 24kHz
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        b64_chunk = writer.write_chunk(chunk)
        if b64_chunk:
            print(f"Chunk {i // chunk_size + 1}: {len(b64_chunk)} Base64 chars")

    # Finalize
    final = writer.write_chunk(finalize=True)
    if final:
        print(f"Final chunk: {len(final)} Base64 chars")

    # Example 2: One-shot conversion with data URI
    print("\n=== One-Shot Base64 with Data URI ===")
    b64_with_uri = audio_to_base64(
        audio, format="mp3", sample_rate=sample_rate, include_data_uri=True
    )
    print(f"Data URI prefix: {b64_with_uri[:50]}...")
    print(f"Total length: {len(b64_with_uri)} chars")

    # Example 3: Raw bytes mode (original behavior)
    print("\n=== Raw Bytes Mode (Original) ===")
    writer_raw = StreamingAudioWriter(
        format="mp3",
        sample_rate=sample_rate,
        channels=1,
        base64_encode=False,  # Original behavior
    )
    raw_data = writer_raw.write_chunk(audio)
    final_raw = writer_raw.write_chunk(finalize=True)
    print(f"Raw bytes: {len(raw_data) + len(final_raw)} bytes")
