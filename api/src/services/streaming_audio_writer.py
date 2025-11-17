"""Audio conversion service with proper streaming support"""
import struct
from io import BytesIO
from typing import Optional
import av
import numpy as np
from loguru import logger


class StreamingAudioWriter:
    """Handles streaming audio format conversions"""
    
    def __init__(self, format: str, sample_rate: int, channels: int = 1):
        self.format = format.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.bytes_written = 0
        self.pts = 0
        self.header_written = False
        
        codec_map = {
            "wav": "pcm_s16le",
            "mp3": "libmp3lame",  # Changed from "mp3" to "libmp3lame"
            "opus": "libopus",
            "flac": "flac",
            "aac": "aac",
        }
        
        if self.format in ["wav", "flac", "mp3", "pcm", "aac", "opus"]:
            if self.format == "pcm":
                # PCM doesn't need encoding
                pass
            else:
                self.output_buffer = BytesIO()
                container_options = {}
                
                # MP3-specific settings for streaming
                if self.format == 'mp3':
                    container_options = {
                        'write_xing': '0',  # Disable VBR header
                    }
                    logger.debug("MP3 streaming mode: disabling Xing VBR header")
                
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
                
                # Codec-specific settings
                if self.format == 'mp3':
                    self.stream.bit_rate = 128000
                    self.stream.options = {
                        'reservoir': '0',  # Disable bit reservoir for better streaming
                    }
                elif self.format in ['aac', 'opus']:
                    self.stream.bit_rate = 128000
                    
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def close(self):
        if hasattr(self, "container"):
            try:
                self.container.close()
            except Exception as e:
                logger.warning(f"Error closing container: {e}")
        if hasattr(self, "output_buffer"):
            self.output_buffer.close()
    
    def write_chunk(
        self, audio_data: Optional[np.ndarray] = None, finalize: bool = False
    ) -> bytes:
        """Write a chunk of audio data and return bytes in the target format.
        
        Args:
            audio_data: Audio data to write, or None if finalizing
            finalize: Whether this is the final write to close the stream
        """
        
        if finalize:
            if self.format != "pcm":
                try:
                    # Flush encoder
                    packets = self.stream.encode(None)
                    for packet in packets:
                        self.container.mux(packet)
                    
                    # Get final data before closing
                    data = self.output_buffer.getvalue()
                    self.close()
                    return data
                except Exception as e:
                    logger.error(f"Error finalizing stream: {e}")
                    self.close()
                    return b""
            return b""
        
        if audio_data is None or len(audio_data) == 0:
            return b""
        
        if self.format == "pcm":
            # Write raw PCM bytes
            return audio_data.tobytes()
        
        try:
            # Create audio frame
            frame = av.AudioFrame.from_ndarray(
                audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.channels == 1 else "stereo",
            )
            frame.sample_rate = self.sample_rate
            frame.pts = self.pts
            self.pts += frame.samples
            
            # Encode frame
            packets = self.stream.encode(frame)
            
            # Get current position before muxing
            current_pos = self.output_buffer.tell()
            
            # Mux packets
            for packet in packets:
                self.container.mux(packet)
            
            # Get only the new data that was written
            self.output_buffer.seek(current_pos)
            new_data = self.output_buffer.read()
            
            # Move to end for next write
            self.output_buffer.seek(0, 2)
            
            return new_data
            
        except Exception as e:
            logger.error(f"Error encoding audio chunk: {e}")
            return b""


class StreamingMP3Writer:
    """Specialized MP3 writer with frame-by-frame encoding"""
    
    def __init__(self, sample_rate: int = 24000, channels: int = 1, bitrate: int = 128000):
        self.sample_rate = sample_rate
        self.channels = channels
        self.bitrate = bitrate
        self.pts = 0
        
        # Create in-memory container
        self.buffer = BytesIO()
        self.container = av.open(self.buffer, mode='w', format='mp3')
        
        # Add audio stream with libmp3lame codec
        self.stream = self.container.add_stream('libmp3lame', rate=sample_rate)
        self.stream.bit_rate = bitrate
        self.stream.layout = 'mono' if channels == 1 else 'stereo'
        
        # Settings for better streaming
        self.stream.options = {
            'reservoir': '0',  # Disable bit reservoir
        }
        
        self.last_position = 0
        
    def write_chunk(self, audio_data: np.ndarray) -> bytes:
        """Write audio chunk and return encoded MP3 data"""
        
        if audio_data is None or len(audio_data) == 0:
            return b""
        
        # Ensure correct shape
        if audio_data.ndim == 1:
            audio_data = audio_data.reshape(1, -1)
        
        # Create frame
        frame = av.AudioFrame.from_ndarray(audio_data, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        self.pts += frame.samples
        
        # Encode
        packets = self.stream.encode(frame)
        
        # Mux packets
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        
        # Get new data
        current_pos = self.buffer.tell()
        self.buffer.seek(self.last_position)
        chunk_data = self.buffer.read(current_pos - self.last_position)
        self.last_position = current_pos
        
        return chunk_data
    
    def finalize(self) -> bytes:
        """Flush encoder and return final data"""
        
        # Flush encoder
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        
        # Get remaining data
        current_pos = self.buffer.tell()
        self.buffer.seek(self.last_position)
        final_data = self.buffer.read()
        
        self.container.close()
        return final_data
