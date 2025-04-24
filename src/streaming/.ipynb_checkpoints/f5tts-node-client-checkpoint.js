const fs = require('fs');
const path = require('path');
// Ensure node-fetch v2 is installed for CommonJS, or configure v3 imports if needed.
// npm install node-fetch@2
const fetch = require('node-fetch');
const { Readable } = require('stream');
// npm install speaker
const Speaker = require('speaker');
// npm install form-data
const FormData = require('form-data');
// npm install wav
const wav = require('wav'); // For robust WAV stream parsing

/**
 * Client for the F5TTS Streaming API
 */
class F5TTSClient {
  /**
   * Create a new F5TTS client
   * @param {Object} options - Client options
   * @param {string} [options.apiUrl='http://localhost:6008'] - URL of the F5TTS API server (updated port)
   * @param {string} [options.defaultVoice='male'] - Default voice reference ID
   */
  constructor(options = {}) {
    this.apiUrl = options.apiUrl || 'http://localhost:6008'; // Default port updated
    this.defaultVoice = options.defaultVoice || 'male';
    this.defaultConfig = {
      nfe_step: 32,
      cfg_strength: 2.0,
      speed: 1.0,
      cross_fade_duration: 0.15, // Note: server-side cross-fade between its internal chunks
      sway_sampling_coef: -1.0
    };
    this.activeSpeaker = null; // To manage the current speaker instance
    this.activeWavReader = null; // To manage the wav reader instance
    this.activeResponseStream = null; // To manage the raw response stream
  }

  /**
   * Helper to handle fetch errors more gracefully
   */
  async _handleFetchError(response) {
    const status = response.status;
    let errorText = `API responded with status ${status}`;
    let errorDetail = '';
    try {
      // Try to parse FastAPI/JSON error detail
      const errorJson = await response.json();
      errorDetail = errorJson.detail || JSON.stringify(errorJson);
    } catch (e) {
      // Fallback to text if not JSON
      try {
        errorDetail = await response.text();
      } catch (e2) {
         errorDetail = "(Could not read error response body)";
      }
    }
    return new Error(`${errorText}: ${errorDetail}`);
  }

  /**
   * Get available voice references
   * @returns {Promise<Object>} - Available voices { id: { name: string, status: boolean|string } }
   */
  async getVoices() {
    console.log(`Fetching voices from ${this.apiUrl}/references`);
    try {
      const response = await fetch(`${this.apiUrl}/references`);
      if (!response.ok) {
        throw await this._handleFetchError(response);
      }
      return await response.json();
    } catch (error) {
      console.error(`Error fetching voices: ${error.message}`);
      throw error;
    }
  }

  /**
   * Check API health
   * @returns {Promise<Object>} - Health status { status: string, message?: string }
   */
  async checkHealth() {
    console.log(`Checking health at ${this.apiUrl}/health`);
    try {
      const response = await fetch(`${this.apiUrl}/health`);
      // Health check might return non-200 but still valid JSON (like warning)
      const data = await response.json();
      if (!response.ok) {
          console.warn(`API health check responded with status ${response.status}`);
      }
      return data; // Return parsed data regardless of status for health check
    } catch (error) {
      console.error(`Error checking API health: ${error.message}`);
      // Provide a structured error for connection issues
       return { status: 'error', message: `Connection failed: ${error.message}` };
    }
  }

  /**
   * Stream TTS audio from F5TTS API and optionally play or save it.
   * Uses the 'speaker' parameter to select the reference voice.
   * @param {Object} options - TTS options
   * @param {string} options.text - Text to synthesize
   * @param {string} [options.voice] - Voice reference ID (overrides defaultVoice)
   * @param {number} [options.nfeStep] - Number of function evaluation steps
   * @param {number} [options.cfgStrength] - Classifier-free guidance strength
   * @param {number} [options.speed] - Speech speed
   * @param {number} [options.crossFadeDuration] - Server-side cross-fade duration
   * @param {number} [options.swayCoef] - Sway sampling coefficient
   * @param {string|null} [options.output=null] - Path to save output audio (e.g., './output/audio.wav'). If null, plays through speakers.
   * @returns {Promise<string|void>} - Output path if saved, void if played successfully. Rejects on error.
   */
  async streamTTS(options) {
    // Merge with default options
    const config = {
      ...this.defaultConfig,
      speaker: options.voice || this.defaultVoice, // Use specific voice or client default
      ...options // Apply specific options, overriding defaults/voice if needed
    };

    // --- Build Request Data (Server expects only 'speaker' for reference) ---
    const requestData = {
      text: config.text,
      speaker: config.speaker, // This is the reference ID the server uses
      nfe_step: config.nfeStep !== undefined ? config.nfeStep : this.defaultConfig.nfe_step,
      cfg_strength: config.cfgStrength !== undefined ? config.cfgStrength : this.defaultConfig.cfg_strength,
      speed: config.speed !== undefined ? config.speed : this.defaultConfig.speed,
      cross_fade_duration: config.crossFadeDuration !== undefined ? config.crossFadeDuration : this.defaultConfig.cross_fade_duration,
      sway_sampling_coef: config.swayCoef !== undefined ? config.swayCoef : this.defaultConfig.sway_sampling_coef
    };
    // --- End Build Request Data ---

    if (!requestData.text || !requestData.text.trim()) {
         throw new Error("Text cannot be empty.");
    }
    if (!requestData.speaker) {
         throw new Error("Speaker (voice reference ID) must be specified.");
    }

    console.log(`Synthesizing speech with voice '${requestData.speaker}'...`);

    try {
      // Make streaming request
      const response = await fetch(`${this.apiUrl}/tts/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'audio/wav' },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw await this._handleFetchError(response);
      }

      if (!response.body) {
        throw new Error("API response did not contain a readable stream body.");
      }

      // Store the raw response stream for potential cleanup
      this.activeResponseStream = response.body;

      // --- Process the response stream ---

      if (config.output) {
        // --- Save to file ---
        const outputPath = path.resolve(config.output); // Ensure absolute path
        const outputDir = path.dirname(outputPath);

        // Ensure directory exists
        try {
            if (!fs.existsSync(outputDir)) {
                console.log(`Creating directory: ${outputDir}`);
                fs.mkdirSync(outputDir, { recursive: true });
            }
        } catch(dirError) {
            throw new Error(`Failed to create output directory '${outputDir}': ${dirError.message}`);
        }


        console.log(`Saving audio stream to: ${outputPath}`);
        const fileStream = fs.createWriteStream(outputPath);

        // Pipe the response body directly to the file stream
        this.activeResponseStream.pipe(fileStream);

        return new Promise((resolve, reject) => {
          // Success condition
          fileStream.on('finish', () => {
            console.log(`Audio successfully saved to ${outputPath}`);
            this.activeResponseStream = null; // Clear reference
            resolve(outputPath);
          });

          // Handle errors during file writing
          fileStream.on('error', (err) => {
            console.error(`Error writing audio file '${outputPath}': ${err.message}`);
            this.activeResponseStream = null; // Clear reference
            reject(err);
          });

          // Handle errors on the source stream (network etc.)
          this.activeResponseStream.on('error', (err) => {
              console.error(`Error reading response stream while saving file: ${err.message}`);
              fileStream.close(); // Ensure file stream is closed
              // Attempt to remove potentially corrupted partial file
              try { if(fs.existsSync(outputPath)) fs.unlinkSync(outputPath); } catch (unlinkErr) {/* ignore */}
              this.activeResponseStream = null; // Clear reference
              reject(err);
          });
        });

      } else {
        // --- Play through speakers ---
        console.log("Streaming audio to speakers...");

        // Stop any existing playback *before* starting the new one
        this.stopPlayback();

        return new Promise((resolve, reject) => {
            // Use wav.Reader to parse the header and output raw PCM
            const wavReader = new wav.Reader();
            this.activeWavReader = wavReader; // Store for potential cleanup

            wavReader.on('format', (format) => {
                console.log(`Audio Format Received: ${format.channels}ch, ${format.sampleRate}Hz, ${format.bitDepth}bit, PCM: ${format.audioFormat === 1}`);
                // Validate format if necessary (Speaker library is generally flexible)
                if (format.audioFormat !== 1) { // 1 = PCM
                     const formatError = new Error(`Unsupported WAV audio format received: ${format.audioFormat}. Expected PCM (1).`);
                     this.stopPlayback(); // Clean up
                     return reject(formatError);
                }

                 try {
                    // Create the speaker instance with the exact format from the header
                    this.activeSpeaker = new Speaker(format);

                     // Handle speaker errors (e.g., device unavailable)
                    this.activeSpeaker.on('error', (err) => {
                         console.error("Speaker error:", err.message);
                         // Don't call stopPlayback here, as it might trigger infinite loop if speaker closing causes error
                         this.activeSpeaker = null; // Clear speaker ref
                         this.activeWavReader = null; // Clear wav ref
                         reject(err); // Reject the main promise
                    });

                    // Handle speaker closing naturally (end of stream)
                    this.activeSpeaker.on('close', () => {
                         console.log("Speaker closed (playback finished).");
                         // Playback completed successfully
                         this.activeSpeaker = null; // Clear ref
                         this.activeWavReader = null; // Clear ref
                         resolve(); // Resolve the main promise
                    });

                    // Pipe the raw PCM data from the wavReader to the speaker
                    console.log("Piping parsed WAV data to speaker...");
                    this.activeWavReader.pipe(this.activeSpeaker);

                 } catch(speakerInitError) {
                     console.error("Failed to initialize Speaker:", speakerInitError.message);
                     this.stopPlayback(); // Clean up reader/response stream
                     reject(speakerInitError);
                 }
            });

            // Handle errors during WAV parsing (e.g., corrupted header)
            wavReader.on('error', (err) => {
                console.error("WAV parsing error:", err.message);
                this.stopPlayback(); // Clean up speaker/response stream
                reject(err);
            });

            // Pipe the raw response body from fetch into the wav Reader
            this.activeResponseStream.pipe(wavReader);

             // Handle errors on the original response stream (network etc.)
             this.activeResponseStream.on('error', (err) => {
                 console.error(`Error reading response stream: ${err.message}`);
                 // Don't call stopPlayback here to avoid potential race conditions/loops
                 // Just reject the promise, cleanup should happen in finally block or on error
                 this.activeWavReader = null;
                 this.activeSpeaker = null;
                 this.activeResponseStream = null;
                 reject(err);
             });

             // Handle premature closing of the response stream
             this.activeResponseStream.on('close', () => {
                console.log("Source response stream closed.");
                // This might happen normally at the end, or if connection drops.
                // The 'close' event on the speaker usually handles the success case.
             });
        });
      }
    } catch (error) {
      console.error(`Error setting up TTS stream: ${error.message}`);
      this.stopPlayback(); // Ensure cleanup on setup error
      this.activeResponseStream = null; // Clear response stream ref too
      throw error; // Re-throw the error for the caller
    }
  }

 /**
  * Stops the current audio playback if active.
  * Ends the Speaker, WavReader, and tries to destroy the response stream.
  */
 stopPlayback() {
     if (this.activeSpeaker) {
         console.log("Stopping active speaker...");
         try {
             this.activeSpeaker.end(); // Gracefully end the speaker stream
         } catch (e) { console.error("Error ending speaker:", e.message); }
         this.activeSpeaker = null;
     }
     if (this.activeWavReader) {
        console.log("Stopping active WAV reader...");
        try {
            // Unpiping might be necessary if stopping forcefully mid-stream
            if (this.activeResponseStream) this.activeResponseStream.unpipe(this.activeWavReader);
            this.activeWavReader.destroy(); // Forcefully stop the reader
        } catch (e) { console.error("Error stopping WAV reader:", e.message); }
        this.activeWavReader = null;
     }
      if (this.activeResponseStream) {
        console.log("Stopping active response stream...");
        try {
            this.activeResponseStream.destroy(); // Destroy the underlying network stream
        } catch(e) { console.error("Error destroying response stream:", e.message); }
         this.activeResponseStream = null;
      }
 }


  /**
   * Upload a custom reference audio file to the server
   * @param {Object} options - Upload options
   * @param {string} options.audioPath - Path to the audio file
   * @param {string} [options.text] - Transcript of the audio (optional but recommended)
   * @returns {Promise<Object>} - Upload result { status: string, message: string, ref_id: string, estimated_name?: string }
   */
  async uploadReference(options) {
    const { audioPath, text } = options;
    if (!audioPath || !fs.existsSync(audioPath)) {
      throw new Error(`Reference audio file not found: ${audioPath}`);
    }

    console.log(`Uploading reference audio: ${path.basename(audioPath)}`);
    if (text) console.log(`With transcript: "${text.substring(0, 50)}..."`);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('file', fs.createReadStream(audioPath), {
          filename: path.basename(audioPath) // Provide filename hint for server logging
      });
      if (text) {
        formData.append('text', text);
      }

      // Upload the file
      const response = await fetch(`${this.apiUrl}/upload_reference`, {
        method: 'POST',
        body: formData,
        // Headers are set automatically by node-fetch with FormData
      });

      if (!response.ok) {
        // Use the error handler to get details
        throw await this._handleFetchError(response);
      }

      const result = await response.json();
      console.log(`Upload successful: ${result.message} (ID: ${result.ref_id})`);
      return result; // Contains status, message, ref_id, estimated_name

    } catch (error) {
      console.error(`Error uploading reference audio: ${error.message}`);
      throw error;
    }
  }
}

// --- Example Usage ---
async function main() {
  const client = new F5TTSClient({
    apiUrl: 'http://localhost:6008', // Ensure port matches server
    defaultVoice: 'male'
  });

  // Create output dir if it doesn't exist for saving examples
  const outputDir = './output';
  if (!fs.existsSync(outputDir)) {
    try {
      console.log(`Creating output directory: ${outputDir}`);
      fs.mkdirSync(outputDir);
    } catch(e) {
      console.error(`Failed to create output directory: ${e.message}`);
      // Decide if execution should stop
      // return;
    }
  }

  try {
    // 1. Check server health
    console.log("\n--- 1. Checking server health ---");
    const health = await client.checkHealth();
    console.log('Server health:', health);
    // Allow 'warning' status (model loaded, no voices ready) but fail on 'error'
    if (health.status === 'error') {
      console.error(`Server is not healthy (${health.message || 'Unknown reason'})! Exiting example.`);
      return;
    }

    // 2. Get available voices
    console.log("\n--- 2. Getting available voices ---");
    const voices = await client.getVoices();
    const voiceIds = Object.keys(voices);
     if (voiceIds.length > 0) {
        console.log('Available voices:', voiceIds.map(id => {
            const status = voices[id].status === true ? 'Ready' : `(${voices[id].status})`;
            return `${id} ('${voices[id].name || 'Unnamed'}' - ${status})`;
        }).join(', '));
    } else {
         console.warn("No voices seem to be loaded or ready on the server.");
         // Allow continuing, but TTS might fail if default 'male' isn't actually ready
    }


    // 3. Simple playback with default voice ('male')
    console.log("\n--- 3. Simple playback (default voice) ---");
    try {
        await client.streamTTS({
        text: 'Chào mừng bạn đến với bản demo máy khách nút js cho F5TTS.',
        // output: null (default) -> play audio
        });
        console.log("Playback 1 finished.");
    } catch(e) {
         console.error(`Playback 1 failed: ${e.message}`);
         // Continue example even if one fails
    }
    await new Promise(resolve => setTimeout(resolve, 300)); // Short pause between tests


    // 4. Save to file with a different voice and custom parameters
    // Find a ready female voice, or the first ready voice if 'female' not found/ready
    let targetVoiceId = 'female';
    if (!voices[targetVoiceId] || voices[targetVoiceId].status !== true) {
        console.log("Female voice not ready, searching for alternative...");
        targetVoiceId = voiceIds.find(id => voices[id].status === true) || null;
    }

    if (targetVoiceId) {
        console.log(`\n--- 4. Save to file (voice: ${targetVoiceId}, faster) ---`);
        try {
            const outputPath = path.join(outputDir, 'node-client-output.wav');
            await client.streamTTS({
              text: 'Âm thanh này được tạo với tốc độ nhanh hơn và lưu vào tệp.',
              voice: targetVoiceId,
              speed: 1.3,
              nfeStep: 28,
              cfgStrength: 1.8,
              output: outputPath
            });
            console.log(`Saved audio successfully.`);
        } catch (e) {
            console.error(`Save to file failed: ${e.message}`);
        }
    } else {
        console.log("\n--- 4. Skipping save-to-file example (no suitable *ready* voice found) ---");
    }
    await new Promise(resolve => setTimeout(resolve, 300)); // Short pause


    /*
    // 5. Example: Upload and use custom reference (Uncomment and UPDATE PATHS to test)
    console.log("\n--- 5. Uploading custom reference (example) ---");
    const customAudioPath = './female-vts.wav'; // !!! <--- UPDATE THIS PATH to a real audio file !!!
    const customAudioText = "Ai đã đến Hàng Dương, đều không thể cầm lòng về những nấm mộ chen nhau, nhấp nhô trải khắp một vùng đồi."; // !!! <--- UPDATE TRANSCRIPT if needed !!!

    if (fs.existsSync(customAudioPath)) {
        try {
            const uploadResult = await client.uploadReference({
                audioPath: customAudioPath,
                text: customAudioText
            });
            const customVoiceId = uploadResult.ref_id;

            console.log(`Upload submitted. Waiting ~10 seconds for server processing (ID: ${customVoiceId})...`);
            await new Promise(resolve => setTimeout(resolve, 10000)); // Allow time for background processing

            // Verify the voice is ready now (optional but good practice)
            const updatedVoices = await client.getVoices();
            if (updatedVoices[customVoiceId] && updatedVoices[customVoiceId].status === true) {
                console.log(`\nSynthesizing with custom voice ID: ${customVoiceId}`);
                const customOutputPath = path.join(outputDir, 'custom-voice-node.wav');
                await client.streamTTS({
                    text: 'Giọng nói này được tạo ra bằng cách sử dụng tệp tham chiếu tùy chỉnh đã tải lên.',
                    voice: customVoiceId, // Use the new ID
                    output: customOutputPath
                });
                console.log(`Saved custom voice synthesis to: ${customOutputPath}`);
            } else {
                 console.warn(`Custom voice ${customVoiceId} not found or not ready after waiting. Status: ${updatedVoices[customVoiceId]?.status}`);
                 console.log("Skipping synthesis with custom voice.");
            }

        } catch (uploadError) {
            console.error("Failed to upload or use custom reference:", uploadError.message);
            console.log("(Check the audio file path and server permissions for './references')");
        }
    } else {
        console.log(`Skipping custom reference test: File not found at '${customAudioPath}'`);
    }
    */

  } catch (error) {
    console.error('\n--- Error in main execution ---');
    // Check if it's a fetch error which might already be detailed
    if (!(error instanceof fetch.FetchError)) {
        console.error(error.message);
        // console.error(error.stack); // Uncomment for more details
    }
    console.error('---------------------------------');
  } finally {
      // Ensure any active playback is stopped when the main function finishes or errors
      console.log("\n--- Example finished, stopping any active playback ---");
      client.stopPlayback();
  }
}

// Run the example if this script is executed directly
if (require.main === module) {
  main();
}

module.exports = { F5TTSClient }; // Export class for potential reuse