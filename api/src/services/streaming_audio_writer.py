"""Audio conversion service with proper streaming support"""
import struct
from io import BytesIO
from typing import Optional
import av
import numpy as np
import soundfile as sf
from loguru import logger
from pydub import AudioSegment


class StreamingAudioWriter:
    """Handles streaming audio format conversions with headers in all chunks"""
    
    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0
        self.header_written = False
        self.cached_header = b""
        
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
                if self.format == 'mp3':
                    # Disable Xing VBR header
                    container_options = {'write_xing': '0'}
                    logger.debug("Disabling Xing VBR header for MP3 encoding.")
                
                self.container = av.open(
                    self.output_buffer,
                    mode="w",
                    format=self.format if self.format != "aac" else "adts",
                    options=container_options
                )
                
                self.stream = self.container.add_stream(
                    codec_map[self.format],
                    rate=self.sample_rate,
                    layout="mono" if self.channels == 1 else "stereo",
                )
                
                # Set bit_rate only for codecs where it's applicable and useful
                if self.format in ['mp3', 'aac', 'opus']:
                    self.stream.bit_rate = 160000
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def close(self):
        if hasattr(self, "container"):
            self.container.close()
        if hasattr(self, "output_buffer"):
            self.output_buffer.close()
    
    def _extract_header(self, data: bytes) -> bytes:
        """Extract header information from the first chunk of encoded data"""
        if self.format == "mp3":
            # For MP3, we need to preserve the initial ID3 tags and first frame header
            # MP3 frames start with 0xFF 0xFB (or similar sync pattern)
            # We'll cache everything before the first actual audio frame
            header_end = 0
            if data.startswith(b'ID3'):
                # Skip ID3v2 tag if present
                if len(data) >= 10:
                    # ID3v2 size is synchsafe integer at bytes 6-9
                    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
                    header_end = 10 + size
            
            # Find first MP3 frame sync (0xFF 0xFx)
            for i in range(header_end, len(data) - 1):
                if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
                    # Found first frame, include frame header in cached header
                    # MP3 frame header is 4 bytes
                    return data[:i + 4] if i + 4 <= len(data) else data[:i]
            
            return data[:header_end] if header_end > 0 else b""
        
        elif self.format == "aac":
            # For AAC with ADTS, each frame has its own header, so no global header needed
            # ADTS frames are self-contained
            return b""
        
        elif self.format == "opus":
            # Opus in Ogg container - extract Ogg page headers
            # First page contains OpusHead, second contains OpusTags
            if data.startswith(b'OggS'):
                pages_found = 0
                pos = 0
                header_end = 0
                
                while pos < len(data) - 27 and pages_found < 2:  # Need at least 27 bytes for Ogg header
                    if data[pos:pos+4] == b'OggS':
                        # Parse Ogg page structure
                        segment_count = data[pos + 26] if pos + 26 < len(data) else 0
                        if pos + 27 + segment_count <= len(data):
                            segment_table = data[pos + 27:pos + 27 + segment_count]
                            page_size = 27 + segment_count + sum(segment_table)
                            header_end = pos + page_size
                            pos = header_end
                            pages_found += 1
                        else:
                            break
                    else:
                        break
                
                return data[:header_end] if pages_found == 2 else b""
            return b""
        
        elif self.format == "flac":
            # FLAC has a streaminfo block at the beginning
            if data.startswith(b'fLaC'):
                # Read through metadata blocks
                pos = 4
                while pos < len(data):
                    if pos + 4 > len(data):
                        break
                    is_last = (data[pos] & 0x80) != 0
                    block_size = ((data[pos] & 0x7F) << 16) | (data[pos + 1] << 8) | data[pos + 2]
                    pos += 4 + block_size
                    if is_last:
                        return data[:pos]
                return data[:pos] if pos < len(data) else b""
            return b""
        
        elif self.format == "wav":
            # WAV header is 44 bytes typically
            if data.startswith(b'RIFF'):
                # Find 'data' chunk
                pos = 12  # Skip RIFF header
                while pos < len(data) - 8:
                    chunk_id = data[pos:pos+4]
                    chunk_size = struct.unpack('<I', data[pos+4:pos+8])[0]
                    if chunk_id == b'data':
                        return data[:pos+8]  # Include up to and including data chunk header
                    pos += 8 + chunk_size
            return b""
        
        return b""
    
    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.
        
        All chunks will include necessary header information to be independently decodable.
        
        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        if finalize:
            if self.format != "pcm":
                # Flush stream encoder
                packets = self.stream.encode(None)
                for packet in packets:
                    self.container.mux(packet)
                
                logger.debug("Muxed final packets.")
                
                # Get the final bytes from the buffer *before* closing it
                data = self.output_buffer.getvalue()
                self.close()
                
                # For final chunk, prepend header if needed
                if self.cached_header and len(data) > 0:
                    return self.cached_header + data
                return data
            else:
                return b""
        
        if audio_data is None or len(audio_data) == 0:
            return b""
        
        if self.format == "pcm":
            # PCM doesn't need headers - raw audio data
            return audio_data.tobytes()
        else:
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
            
            data = self.output_buffer.getvalue()
            self.output_buffer.seek(0)
            self.output_buffer.truncate(0)
            
            # Extract and cache header from first chunk
            if not self.header_written and len(data) > 0:
                self.cached_header = self._extract_header(data)
                self.header_written = True
                logger.debug(f"Cached header for {self.format}: {len(self.cached_header)} bytes")
                return data  # First chunk already has header
            
            # For subsequent chunks, prepend the cached header
            if self.cached_header and len(data) > 0:
                return self.cached_header + data
            
            return data
