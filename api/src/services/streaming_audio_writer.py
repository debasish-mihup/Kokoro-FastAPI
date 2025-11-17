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
    
    def _needs_header_per_chunk(self) -> bool:
        """Determine if this format needs headers in every chunk for independent decoding"""
        # ADTS AAC has headers per frame already
        # MP3, WAV, FLAC, Opus need special handling
        return self.format in ["aac"]  # Only ADTS AAC is truly self-describing per frame
    
    def _extract_header(self, data: bytes) -> bytes:
        """Extract header information from the first chunk of encoded data
        
        For most formats, this extracts metadata/codec info that should only 
        appear at the start of the stream. For ADTS AAC, returns empty as each
        frame is self-contained.
        """
        if self.format == "mp3":
            # For MP3, extract only ID3 tags (if present)
            # Don't include frame headers as they're part of the frame data
            header_end = 0
            if data.startswith(b'ID3'):
                # Skip ID3v2 tag if present
                if len(data) >= 10:
                    # ID3v2 size is synchsafe integer at bytes 6-9
                    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
                    header_end = 10 + size
                    return data[:header_end]
            return b""
        
        elif self.format == "aac":
            # For AAC with ADTS, each frame has its own header, so no global header needed
            # ADTS frames are self-contained - this is the format that works best for streaming
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
        
        Each chunk will contain necessary header information to be valid:
        - MP3: ID3 tag (if present) prepended to each chunk
        - AAC: ADTS frames already self-contained with headers
        - WAV: RIFF header prepended to each chunk with updated size
        - Other formats: Headers prepended where applicable
        
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
                
                # For final chunk, apply header if needed
                if len(data) > 0:
                    return self._prepare_chunk_with_header(data, is_final=True)
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
                if len(self.cached_header) > 0:
                    logger.debug(f"Cached header for {self.format}: {len(self.cached_header)} bytes")
                else:
                    logger.debug(f"No extractable header for {self.format} (format has per-frame headers)")
            
            # Prepare chunk with appropriate header
            if len(data) > 0:
                return self._prepare_chunk_with_header(data, is_final=False)
            
            return data
    
    def _prepare_chunk_with_header(self, chunk_data: bytes, is_final: bool = False) -> bytes:
        """Prepare a chunk with appropriate header for the format
        
        Args:
            chunk_data: The encoded chunk data
            is_final: Whether this is the final chunk
            
        Returns:
            Chunk data with header prepended if applicable
        """
        if self.format == "aac":
            # ADTS AAC frames already contain headers - return as-is
            return chunk_data
        
        elif self.format == "mp3":
            # For MP3, prepend ID3 tag (if we have one) to make chunk more compatible
            # Each chunk will be: [ID3 tag (optional)] + [MP3 frames]
            if self.cached_header and len(self.cached_header) > 0:
                # Only prepend ID3, not frame headers
                return self.cached_header + chunk_data
            return chunk_data
        
        elif self.format == "wav":
            # For WAV, create a complete RIFF/WAVE header for this chunk
            # This makes each chunk a valid WAV file
            if not self.cached_header:
                return chunk_data
            
            # Strip any existing RIFF header from chunk_data
            audio_data = chunk_data
            if chunk_data.startswith(b'RIFF'):
                # Find the 'data' chunk and extract only audio data
                pos = 12
                while pos < len(chunk_data) - 8:
                    chunk_id = chunk_data[pos:pos+4]
                    chunk_size = struct.unpack('<I', chunk_data[pos+4:pos+8])[0]
                    if chunk_id == b'data':
                        audio_data = chunk_data[pos+8:pos+8+chunk_size]
                        break
                    pos += 8 + chunk_size
            
            # Create new WAV header with correct data size
            data_size = len(audio_data)
            file_size = 36 + data_size  # Total file size - 8
            
            # Build complete WAV header
            wav_header = b'RIFF'
            wav_header += struct.pack('<I', file_size)
            wav_header += b'WAVE'
            wav_header += b'fmt '
            wav_header += struct.pack('<I', 16)  # fmt chunk size
            wav_header += struct.pack('<H', 1)   # audio format (PCM)
            wav_header += struct.pack('<H', self.channels)
            wav_header += struct.pack('<I', self.sample_rate)
            wav_header += struct.pack('<I', self.sample_rate * self.channels * 2)  # byte rate
            wav_header += struct.pack('<H', self.channels * 2)  # block align
            wav_header += struct.pack('<H', 16)  # bits per sample
            wav_header += b'data'
            wav_header += struct.pack('<I', data_size)
            
            return wav_header + audio_data
        
        elif self.format == "flac":
            # For FLAC, prepend the streaminfo and metadata blocks
            if self.cached_header and len(self.cached_header) > 0:
                return self.cached_header + chunk_data
            return chunk_data
        
        elif self.format == "opus":
            # For Opus in Ogg, prepend the header pages (OpusHead + OpusTags)
            if self.cached_header and len(self.cached_header) > 0:
                return self.cached_header + chunk_data
            return chunk_data
        
        else:
            # Unknown format - return as-is
            return chunk_data
