import os
import json
import time
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse # Added HTMLResponse
from pydantic import BaseModel
import asyncio
import io
import wave
from vinorm import TTSnorm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback # For better error logging

# Import F5TTS wrapper
from f5tts_wrapper import F5TTSWrapper

# --- Configuration ---
# Adjust these paths/settings as needed
MODEL_CONFIG = {
    "model_name": "./F5TTS_v1_Custom_Pruned_12.yaml",  # Path to model config YAML
    "vocoder_name": "vocos",                           # Or "bigvgan"
    "ckpt_path": "../F5-TTS/ckpts/steve_combined_multi/model_last.pt", # Path to your .pt or .safetensors checkpoint
    "vocab_file": "/root/18April/F5-TTS/data/steve_combined_multi_char/vocab.txt", # Path to vocab.txt
    "use_ema": False,                                   # Set based on your checkpoint (often False for custom/pruned)
    "target_sample_rate": 24000,                       # Must match vocoder and model training
    "use_duration_predictor": False                    # Set True if your model has one and you want to use it
}

DEFAULT_REFERENCES = {
    "male": {
        "audio": "./male_south_TEACH_chunk_0_segment_684.wav", # Update path
        "text": "Người người hô hào thay đổi phương pháp giảng dạy. Bộ giáo dục và đào tạo Việt Nam không thiếu những dự án nhằm thay đổi diện mạo giáo dục nước nhà. Nhưng trong khi những thành quả đổi mới còn chưa kịp thu về, thì những ví dụ điển hình về bước lùi của giáo dục ngày càng hiện rõ.",
        "name": "Male Voice (South)"
    },
    "female": {
        "audio": "./female-vts.wav", # Update path
        "text": "Ai đã đến Hàng Dương, đều không thể cầm lòng về những nấm mộ chen nhau, nhấp nhô trải khắp một vùng đồi. Những nấm mộ có tên và không tên, nhưng nấm mộ lấp ló trong lùm cây, bụi cỏ.",
        "name": "Female Voice (VTS)"
    }
}

CUSTOM_REF_PATH = "./references"  # Directory to store custom reference files
TEXT_SPLITTER_CHUNK_SIZE = 100     # Adjusted chunk size
TEXT_SPLITTER_CHUNK_OVERLAP = 0   # Adjusted overlap
# --- End Configuration ---

class TTSRequest(BaseModel):
    text: str
    # ref_audio_path: Optional[str] = None # We'll use 'speaker' to select presets or custom IDs
    # ref_text: Optional[str] = None       # Text associated with custom upload
    speaker: Optional[str] = "male"  # Default to 'male', can be 'female' or a custom ref_id
    nfe_step: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15
    sway_sampling_coef: float = -1.0

app = FastAPI(title="F5TTS Streaming API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
tts_model: Optional[F5TTSWrapper] = None
reference_cache: Dict[str, Dict[str, Any]] = {} # Stores info about loaded references
os.makedirs(CUSTOM_REF_PATH, exist_ok=True)

# Initialize text splitter for chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator=True
)

@app.on_event("startup")
async def startup_event():
    """Initialize the F5TTS model on startup"""
    global tts_model, reference_cache
    print("Starting up FastAPI server...")
    try:
        # Initialize the F5TTS wrapper
        print(f"Loading F5TTS model with config: {MODEL_CONFIG}")
        tts_model = F5TTSWrapper(**MODEL_CONFIG)
        print(f"F5TTS model loaded successfully. Device: {tts_model.device}")

        # Pre-load default reference audios
        await load_default_references()

    except Exception as e:
        print(f"FATAL: Failed to load F5TTS model during startup: {e}")
        traceback.print_exc()
        tts_model = None # Ensure model is None if loading failed

async def load_default_references():
    """Load default reference audio files and cache processed data."""
    global reference_cache, tts_model

    if tts_model is None:
        print("Skipping default reference loading: Model not initialized.")
        return

    print("Loading and processing default references for caching...")
    for ref_id, ref_data in DEFAULT_REFERENCES.items():
        processed_data_cached = False
        try:
            audio_path = ref_data["audio"]
            if os.path.exists(audio_path):
                print(f"Processing default reference: {ref_id} from {audio_path}")

                # --- Call preprocess_reference ONCE ---
                # This loads audio, converts, transcribes (if needed), calculates mel
                _ , processed_ref_text = tts_model.preprocess_reference(
                    ref_audio_path=audio_path,
                    ref_text=TTSnorm(ref_data.get("text", "")).strip(),
                    clip_short=False # Use True here for consistent initial processing
                )

                # --- Capture the processed results from the model ---
                # Ensure tensor is detached and potentially moved to CPU if caching large tensors off-GPU is desired
                # For simplicity, let's assume caching on the model's device is acceptable for now.
                cached_mel = tts_model.ref_audio_processed.clone().detach() # Clone to avoid issues if model state changes elsewhere
                cached_mel_len = tts_model.ref_audio_len
                cached_text = tts_model.ref_text # This should be the final processed text

                # --- Store processed data in cache ---
                reference_cache[ref_id] = {
                    "ref_audio_path": audio_path,
                    "ref_text_original": ref_data.get("text", ""), # Store original for reference
                    "loaded": True,
                    "name": ref_data.get("name", ref_id),
                    "processed_mel": cached_mel,
                    "processed_text": cached_text,
                    "processed_mel_len": cached_mel_len,
                    "error": None
                }
                processed_data_cached = True
                print(f"Successfully processed and cached reference '{ref_id}'. Mel shape: {cached_mel.shape}, Len: {cached_mel_len}")

            else:
                print(f"Warning: Default reference audio '{ref_id}' not found at {audio_path}")
                reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": "File not found"}

        except Exception as e:
            print(f"Error processing reference '{ref_id}' for caching: {e}")
            traceback.print_exc()
            reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": str(e)}
        finally:
            # --- IMPORTANT: Reset model state after processing this reference ---
            # Prevents this reference from interfering with the next one during startup loop
            # or subsequent requests if processing failed mid-way.
            if tts_model:
                tts_model.ref_audio_processed = None
                tts_model.ref_text = None
                tts_model.ref_audio_len = None
                # Optional: Clear GPU cache if memory is tight after processing large files
                # if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("Default reference processing and caching complete.")
    

def create_wave_header(sample_rate, num_channels=1, bits_per_sample=16, data_size=0):
    """Create a wave header for streaming. data_size=0 means unknown size."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        # Write dummy data size if 0, otherwise calculate based on actual data
        # For streaming, we often don't know the final size, so we write 0s
        # or a large number, but many players handle 0 correctly.
        # Let's calculate the header size without data size field first
        wf.writeframes(b'') # Write empty frames to finalize header structure

    header_bytes = buffer.getvalue()

    # If data_size is known and > 0, patch the header (positions 4 and 40)
    # Otherwise, return the header assuming unknown size (often works for streaming players)
    if data_size > 0:
         # Recalculate header with known size
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.setnframes(data_size // (num_channels * (bits_per_sample // 8)))
            wf.writeframes(b'') # Finalize with correct frame count
        header_bytes = buffer.getvalue()


    # For streaming, we need the header size minus the initial 8 bytes ('RIFF', size, 'WAVE')
    # But wave module writes the full header. Return the full header.
    return header_bytes


def process_chunk(chunk_text: str, model: F5TTSWrapper, request: TTSRequest) -> Optional[bytes]:
    """Process a single text chunk and return raw audio bytes (int16)"""
    
    # Normalize text
    chunk_text = TTSnorm(chunk_text).strip() # Normalization done before splitting now
    
    # Minor cleanup for potentially odd splits
    chunk_text = chunk_text.strip()
    
    if not chunk_text:
        return None
    if chunk_text.endswith(".."): # Fix potential vinorm artifacts
        chunk_text = chunk_text[:-1].strip()

    if not chunk_text: # Check again after potential stripping
        return None

    # Log the chunk being processed
    print(f"  Synthesizing chunk: '{chunk_text}'")

    # Generate audio for the chunk
    try:
        # Use the model's generate function, requesting numpy output
        audio_array, sample_rate = model.generate(
            text=chunk_text,
            return_numpy=True,
            nfe_step=request.nfe_step,
            cfg_strength=request.cfg_strength,
            speed=request.speed,
            cross_fade_duration=request.cross_fade_duration, # Used internally by generate if needed
            sway_sampling_coef=request.sway_sampling_coef,
            use_duration_predictor=model.use_duration_predictor # Use the model's configured setting
        )

        # Check if generation was successful
        if audio_array is None or audio_array.size == 0:
            print(f"Warning: Model generated empty audio for chunk: '{chunk_text}'")
            return None

        # Convert numpy array (float32) to int16 bytes
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return audio_bytes
    except Exception as e:
        print(f"Error generating audio for chunk '{chunk_text}': {e}")
        traceback.print_exc()
        return None

import os
import time
import numpy as np
from typing import Optional, Dict, Any, AsyncGenerator
from fastapi import HTTPException
from pydantic import BaseModel # Assuming TTSRequest is defined elsewhere
import asyncio
import io
import wave
from vinorm import TTSnorm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter # Assuming defined elsewhere
import traceback

async def stream_audio_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """
    Generates audio chunks using cached reference data.

    Handles reference voice selection, text processing, chunked TTS generation,
    and yields WAV audio data (header first, then PCM chunks).

    Args:
        request: The TTSRequest object containing synthesis parameters.

    Yields:
        Bytes representing parts of the WAV audio stream.

    Raises:
        HTTPException: If the model is not ready, reference is invalid/not cached,
                       text is missing, or an internal error occurs.
    """
    global tts_model, reference_cache, text_splitter # Added text_splitter

    start_time = time.time()
    print(f"[stream_audio_generator] Request received at {start_time:.2f}")

    # --- 1. Check Model Availability ---
    if tts_model is None:
        print("Error: F5TTS model is not initialized.")
        raise HTTPException(status_code=503, detail="TTS model is not ready. Please try again later.")

    # --- 2. Select and Retrieve Cached Reference ---
    selected_ref_id = request.speaker or "male"
    print(f"[stream_audio_generator] Speaker selected: '{selected_ref_id}'")

    cached_ref_data = reference_cache.get(selected_ref_id)

    # Validate cached data existence and readiness
    if (not cached_ref_data or
            cached_ref_data.get("loaded") != True or
            "processed_mel" not in cached_ref_data or
            "processed_text" not in cached_ref_data or
            "processed_mel_len" not in cached_ref_data):

        status = cached_ref_data.get('loaded', 'Not Found') if cached_ref_data else 'Not Found'
        error_msg = cached_ref_data.get('error') if cached_ref_data else 'N/A'
        error_detail = f"Reference speaker '{selected_ref_id}' is not ready. Status: {status}."
        if status == 'processing':
            error_detail += " Still processing, please wait."
        elif error_msg:
             error_detail += f" Error during processing: {error_msg}"

        print(f"Error: Required processed data for reference '{selected_ref_id}' not found in cache. {error_detail}")
        # If it's processing, 503 might be suitable. If failed/missing, 404.
        status_code = 503 if status == 'processing' else 404
        raise HTTPException(status_code=status_code, detail=error_detail)

    # --- 3. Set Model State from Cache (NO Preprocessing Call Here!) ---
    try:
        print(f"[stream_audio_generator] Using cached reference data for '{selected_ref_id}'")
        # Directly assign the cached, processed data to the model instance attributes
        tts_model.ref_audio_processed = cached_ref_data["processed_mel"]
        tts_model.ref_text = cached_ref_data["processed_text"] # Use the processed text from cache
        tts_model.ref_audio_len = cached_ref_data["processed_mel_len"]

        # Optional: Verify tensor device matches model device, move if necessary
        if tts_model.ref_audio_processed.device != tts_model.device:
            print(f"Warning: Cached mel tensor device ({tts_model.ref_audio_processed.device}) differs from model device ({tts_model.device}). Moving tensor.")
            tts_model.ref_audio_processed = tts_model.ref_audio_processed.to(tts_model.device)

        print(f"[stream_audio_generator] Model reference state set from cache.")
        print(f"  Cached Mel Shape: {tts_model.ref_audio_processed.shape}, Len: {tts_model.ref_audio_len}")
        # print(f"  Cached Ref Text Used: '{tts_model.ref_text}'") # Potentially long

    except Exception as e:
        print(f"Error setting model state from cached reference '{selected_ref_id}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error applying cached reference for '{selected_ref_id}'.")

    # --- 4. Process Input Text (Same as before) ---
    input_text = request.text
    # ... (rest of text normalization and splitting logic remains the same) ...
    if not input_text or not input_text.strip():
        print("Error: No text provided in the request.")
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    print(f"[stream_audio_generator] Normalizing input text (first 100 chars): '{input_text[:100]}...'")
    try:
        normalized_text = TTSnorm(input_text).strip()
        print(f"[stream_audio_generator] Normalized text (first 100 chars): '{normalized_text[:100]}...'")
    except Exception as e:
        print(f"Warning: Text normalization failed: {e}. Proceeding with original text.")
        traceback.print_exc()
        normalized_text = input_text.strip()

    print(f"[stream_audio_generator] Splitting text using chunk_size={text_splitter._chunk_size}, overlap={text_splitter._chunk_overlap}")
    text_chunks = text_splitter.split_text(normalized_text)
    num_chunks = len(text_chunks)
    print(f"[stream_audio_generator] Text split into {num_chunks} chunks.")

    if num_chunks == 0:
        print("Warning: Text resulted in zero chunks after splitting.")
        yield create_wave_header(tts_model.target_sample_rate)
        return

    # --- 5. Stream Audio Generation (Same as before, uses model state set from cache) ---
    sample_rate = tts_model.target_sample_rate
    print(f"[stream_audio_generator] Starting audio stream generation at {sample_rate} Hz...")
    try:
        yield create_wave_header(sample_rate=sample_rate, data_size=0)
        print("[stream_audio_generator] WAV header yielded.")
    except Exception as header_e:
        print(f"Error creating/yielding WAV header: {header_e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error creating audio header.")

    total_bytes_yielded = 0
    for i, chunk_text in enumerate(text_chunks):
        # ... (the loop calling process_chunk and yielding bytes remains the same) ...
        chunk_num = i + 1
        chunk_start_time = time.time()
        print(f"[stream_audio_generator] Processing chunk {chunk_num}/{num_chunks}...")
        audio_bytes = process_chunk(chunk_text, tts_model, request) # process_chunk uses the currently set model state

        if audio_bytes and len(audio_bytes) > 0:
            try:
                yield audio_bytes
                bytes_yielded = len(audio_bytes)
                total_bytes_yielded += bytes_yielded
                chunk_duration = time.time() - chunk_start_time
                print(f"  [Chunk {chunk_num}] Yielded {bytes_yielded} bytes. Time: {chunk_duration:.3f}s")
            except Exception as yield_e:
                print(f"Error yielding audio bytes for chunk {chunk_num}: {yield_e}")
                break # Stop generation if client disconnects
        else:
            print(f"  [Chunk {chunk_num}] Skipped yielding (no audio data generated or error in process_chunk).")


    # --- 6. Finalize Stream (Same as before) ---
    final_delay = 0.05
    print(f"[stream_audio_generator] Finished processing all chunks. Adding final delay of {final_delay}s...")
    await asyncio.sleep(final_delay)

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"[stream_audio_generator] Stream generation complete.")
    print(f"  Total audio bytes yielded (excluding header): {total_bytes_yielded}")
    print(f"  Total request processing time: {total_duration:.3f} seconds.")

    # --- 7. IMPORTANT: Reset Model State (Optional but good practice) ---
    # Clear the reference state from the model instance after the request is fully processed.
    # Prevents state potentially leaking if the same model instance handles another request
    # before the generator is fully cleaned up (less likely with FastAPI's request handling, but safer).
    tts_model.ref_audio_processed = None
    tts_model.ref_text = None
    tts_model.ref_audio_len = None
    print("[stream_audio_generator] Cleared model reference state post-request.")
    
@app.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """Stream TTS audio for given text using the selected speaker."""
    try:
        # Set appropriate headers for streaming WAV
        headers = {
            "Content-Type": "audio/wav",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # "X-Content-Duration": Estimate? Probably not reliable.
        }
        return StreamingResponse(
            stream_audio_generator(request),
            media_type="audio/wav",
            headers=headers
        )
    except HTTPException as he:
        # Re-raise HTTPExceptions directly
        raise he
    except Exception as e:
        print(f"Unhandled error in /tts/stream endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error generating audio stream: {str(e)}"
        )

async def process_and_cache_reference(file_path: str, text: Optional[str], ref_id: str):
    """Process a reference audio file in background and cache processed data."""
    global reference_cache, tts_model
    print(f"Background task: Processing reference {ref_id} from {file_path}")

    if tts_model is None:
        print(f"Background task error: Model not loaded. Cannot process reference {ref_id}.")
        reference_cache[ref_id] = {
            "ref_audio_path": file_path,
            "loaded": False,
            "error": "TTS Model not available during processing.",
            "name": f"Custom {ref_id.split('_')[-1]} (Error)"
        }
        return

    processed_data_cached = False
    try:
        # Normalize provided text, or prepare for transcription if none
        normalized_text = TTSnorm(text).strip() if text else ""

        # --- Call preprocess_reference ONCE ---
        _ , processed_ref_text = tts_model.preprocess_reference(
            ref_audio_path=file_path,
            ref_text=normalized_text, # Pass normalized text or empty string
            clip_short=True # Use True here for consistent initial processing
        )

        # --- Capture the processed results ---
        cached_mel = tts_model.ref_audio_processed.clone().detach()
        cached_mel_len = tts_model.ref_audio_len
        cached_text = tts_model.ref_text

        # --- Update cache with processed data ---
        reference_cache[ref_id].update({ # Update existing placeholder entry
            "ref_text_original": text,
            "loaded": True,
            "processed_mel": cached_mel,
            "processed_text": cached_text,
            "processed_mel_len": cached_mel_len,
            "error": None,
            # Keep existing name or update if desired
            # "name": f"Custom {ref_id.split('_')[-1]}"
        })
        processed_data_cached = True
        print(f"Background task: Successfully processed and cached reference '{ref_id}'. Mel shape: {cached_mel.shape}, Len: {cached_mel_len}")

    except Exception as e:
        print(f"Background task error: Processing reference {ref_id} failed: {e}")
        traceback.print_exc()
        # Update cache entry to reflect error
        reference_cache[ref_id].update({
            "loaded": False,
            "error": str(e),
            "name": f"{reference_cache[ref_id].get('name', ref_id)} (Error)"
        })
    finally:
        # --- IMPORTANT: Reset model state after processing ---
        if tts_model:
            tts_model.ref_audio_processed = None
            tts_model.ref_text = None
            tts_model.ref_audio_len = None
            # if torch.cuda.is_available(): torch.cuda.empty_cache()

@app.post("/upload_reference")
async def upload_reference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None) # Make text optional
):
    """Upload a custom reference audio file for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create a unique reference ID
    timestamp = int(time.time())
    ref_id = f"custom_{timestamp}"
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Basic validation for audio types
    if file_extension not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_extension}")

    # Define the path where the file will be saved
    ref_path = os.path.join(CUSTOM_REF_PATH, f"{ref_id}{file_extension}")
    print(f"Receiving file upload: {file.filename}, saving as {ref_path}")

    try:
        # Save the file async
        file_content = await file.read()
        with open(ref_path, "wb") as buffer:
            buffer.write(file_content)
        print(f"File saved successfully: {ref_path}")

        # Add background task to process and cache the reference
        background_tasks.add_task(
            process_and_cache_reference,
            ref_path,
            text, # Pass optional text
            ref_id
        )

        # Immediately add placeholder to cache so UI knows it's coming
        reference_cache[ref_id] = {
            "ref_audio_path": ref_path,
            "loaded": "processing", # Indicate processing state
            "name": f"Custom {timestamp} (Processing...)"
        }

        return {
            "status": "processing",
            "message": "Reference audio uploaded. Processing in background.",
            "ref_id": ref_id,
             "estimated_name": reference_cache[ref_id]["name"]
        }
    except Exception as e:
        # --- THIS IS THE CORRECTED CLEANUP BLOCK ---
        print(f"Error saving/uploading reference file: {e}")
        traceback.print_exc()

        # Attempt to clean up the partially saved file if it exists
        print(f"Attempting cleanup of potentially partial file: {ref_path}")
        try:
            # Check if the file exists before attempting removal
            if os.path.exists(ref_path):
                os.remove(ref_path)
                print(f"Successfully removed potentially partial file: {ref_path}")
            else:
                print(f"File {ref_path} did not exist, no cleanup needed.")
        except OSError as remove_error:
            # Log an error if cleanup fails, but don't prevent the original error report
            print(f"Warning: Failed to remove partially saved file {ref_path} during cleanup: {remove_error}")
        # --- END OF CORRECTED CLEANUP BLOCK ---

        # Raise the original exception that caused the failure
        raise HTTPException(status_code=500, detail=f"Error saving reference file: {str(e)}")
        
@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve the HTML client page."""
    try:
        with open("client.html", "r", encoding="utf-8") as file: # Specify encoding
            return file.read()
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Client HTML file not found.")

@app.get("/references")
async def get_references():
    """Get available reference presets (default and custom)."""
    # Return references that are loaded or currently processing
    available_refs = {
        ref_id: {"name": data.get("name", ref_id), "status": data.get("loaded", "error")}
        for ref_id, data in reference_cache.items()
        if data.get("loaded") == True or data.get("loaded") == "processing"
    }
    return available_refs

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_model is None:
        return {"status": "error", "message": "F5TTS model not initialized"}
    # Maybe add a quick check if references are loaded?
    if not any(data.get("loaded") == True for data in reference_cache.values()):
         return {"status": "warning", "message": "Model loaded, but no reference voices are ready."}
    return {"status": "ok", "message": "F5TTS model ready."}

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    # Consider adding reload=True for development, but remove for production
    uvicorn.run(app, host="0.0.0.0", port=6008)