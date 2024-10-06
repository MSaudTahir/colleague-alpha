import asyncio
# import sys
# import threading
# import time
import discord
import os # default module
from dotenv import load_dotenv
load_dotenv() # load all the variables from the env file
import numpy as np
import pyaudio   
import librosa
import soundfile as sf
from datetime import datetime

# from llm.google_handler import Google_LLM
# from llm.openai_handler import OpenAI_LLM
from custom_sinks import StreamSink

from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import pipeline

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading VAD...", end = " ")
vad_model = load_silero_vad()
print("loaded!")

print("Loading STT...", end = " ")
stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=device)
def stt_transcribe(audio_data):
    audio_data_32 = audio_data / (1 << 15)
    audio_data_32 = audio_data_32.astype(np.float32)
    resampled =  librosa.resample(audio_data_32, orig_sr=48000, target_sr=16000)
    sf.write(open(f"audio_{datetime.now().strftime('%m %d %Y, %H %M %S')}.wav", "wb"), resampled, 16000)
    return stt_pipe(resampled)["text"]
print("loaded!")

print("Loading LLM...", end = " ")
llm_pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct",  torch_dtype=torch.bfloat16, device=device)
def llm_inference(messages):
    outputs = llm_pipe(
        messages,   
        max_new_tokens=1024,
    )
    return outputs[0]["generated_text"][-1]["content"]
print("loaded!")

# print("Loading TTS...", end = " ")
# from melo.api import TTS
# speed = 1.0
# device = 'auto'
# model = TTS(language='EN', device=device)
# speaker_ids = model.hps.data.spk2id
# def tts_generation(text):
#     audio_arr = model.tts_to_file(text, speaker_ids['EN-Default'], None, speed=speed)
#     resampled = librosa.resample(audio_arr, 44100, target_sr=48000)
#     max_16bit = 2**15
#     raw_data = resampled * max_16bit
#     raw_data = raw_data.astype(np.int16)
#     stereo_audio = np.stack((raw_data, raw_data), axis=-1)
#     stereo_audio_flat = stereo_audio.flatten()
#     return stereo_audio_flat
# print("loaded!")

# print("Loading TTS...", end = " ")
# import torch
# from transformers import AutoTokenizer
# from parler_tts import ParlerTTSForConditionalGeneration
# import librosa 
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
# tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
# def tts_generation(text): 
#     description = "A old woman speaker delivers a speech in a raspy voice with low pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
#     input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
#     prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
#     generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
#     audio_arr = generation.cpu().numpy().squeeze()
#     resampled =  librosa.resample(audio_arr, orig_sr=model.config.sampling_rate, target_sr=48000)
#     max_16bit = 2**15
#     raw_data = resampled * max_16bit
#     raw_data = raw_data.astype(np.int16)
#     print(raw_data)
#     stereo_audio = np.stack((raw_data, raw_data), axis=-1)
#     print(stereo_audio)
#     stereo_audio_flat = stereo_audio.flatten()
#     print(stereo_audio_flat)
#     return stereo_audio_flat
# print("loaded!")

print("Loading TTS...", end = " ")
from transformers import VitsTokenizer, VitsModel, set_seed
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
def tts_generation(text): 
   inputs = tokenizer(text=text, return_tensors="pt").to(device)
   set_seed(666)  # make deterministic
   with torch.no_grad():
      outputs = model(**inputs)
   audio_arr = outputs.waveform[0]
   audio_arr = audio_arr.cpu().numpy().squeeze()
   resampled =  librosa.resample(audio_arr, orig_sr=16000, target_sr=48000)
   max_16bit = 2**15
   raw_data = resampled * max_16bit
   raw_data = raw_data.astype(np.int16)
   stereo_audio = np.stack((raw_data, raw_data), axis=-1)
   stereo_audio_flat = stereo_audio.flatten()
   return stereo_audio_flat
print("loaded!")


bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")

@bot.command(description="Sends the bot's latency.") # this decorator makes a slash command
async def ping(ctx: discord.ApplicationContext): # a slash command will be created with the name "ping"
    await ctx.respond(f"Pong! Latency is {bot.latency}")

connections = {}

@bot.command()
async def join_voice_channel(ctx: discord.ApplicationContext):
    voice = ctx.author.voice

    await ctx.response.defer()

    if not voice:
        await ctx.followup.send("You aren't in a voice channel!")

    vc = await voice.channel.connect()  # Connect to the voice channel the author is in.

    connections.update({ctx.guild.id: [vc, voice.channel.name]})  # Updating the cache with the guild and channel.
    await ctx.followup.send(f"Connected to channel {voice.channel.name}")

@bot.command()
async def leave_voice_channel(ctx: discord.ApplicationContext):
    await ctx.response.defer()
    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc: discord.VoiceClient = connections[ctx.guild.id][0]
        vc_name = connections[ctx.guild.id][1]
        try:
            vc.stop_recording()  # Stop recording if doing so
        except discord.sinks.errors.RecordingException:
            pass
        del connections[ctx.guild.id]  # Remove the guild from the cache.
        await ctx.delete()  # And delete.
        await vc.disconnect()
        await ctx.followup.send(f"Left channel {vc_name}")
    else:
        await ctx.followup.send("Not in any voice channel!")

@bot.command()
async def listen_to_server(ctx: discord.ApplicationContext):  # If you're using commands.Bot, this will also work.
    CHUNK_MS = 20
    RATE = 48000
    CHUNK = int(RATE/(1000/CHUNK_MS))
    FORMAT = pyaudio.paInt32
    CHANNELS = 1

    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc: discord.VoiceClient = connections[ctx.guild.id][0]

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK
        )

        await ctx.response.defer()

        while vc.is_connected():
            data = stream.read(CHUNK)
            vc.send_audio_packet(data, encode=True)

        stream.stop_stream()
        stream.close()
        p.terminate()

        await ctx.followup.send("Listen ended.")
    else:
        await ctx.followup.send("Not in any voice channel!")

@bot.command()
async def echo_voice_channel(ctx: discord.ApplicationContext):
    await ctx.response.defer()
    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc: discord.VoiceClient = connections[ctx.guild.id][0]

        sink = StreamSink()

        vc.start_recording(
            sink,  # The sink type to use.
            record_callback,  # What to do once done.
            ctx.channel  # The channel to disconnect from.
        )

        await ctx.followup.send("Echo started.")
        print("Starting to send audio.")

        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paInt16,
                        channels=2,
                        rate=48000,
                        output=True)

        sent_audio_index = 0

        while vc.is_connected():
            if len(sink.audio_data) > 0:
                audio_data = sink.audio_data[[*sink.audio_data.keys()][0]]
                if sent_audio_index < len(audio_data): 
                    stream.write(audio_data[sent_audio_index])
                    vc.send_audio_packet(audio_data[sent_audio_index], encode=True)
                    sent_audio_index += 1
    else:
        await ctx.followup.send("Not in any voice channel!")

@bot.command()
async def join_and_echo(ctx: discord.ApplicationContext):
    await ctx.response.defer()
    voice = ctx.author.voice

    if not voice:
        await ctx.followup.send("You aren't in a voice channel!")

    vc = await voice.channel.connect()  # Connect to the voice channel the author is in.
    connections.update({ctx.guild.id: [vc, voice.channel.name]})  # Updating the cache with the guild and channel.
    await ctx.followup.send(f"Connected to channel {voice.channel.name}")

    sink = StreamSink()

    vc.start_recording(
        sink,  # The sink type to use.
        record_callback,  # What to do once done.
        ctx.channel  # The channel to disconnect from.
    )

    await ctx.followup.send("Echo started.")
    print("Starting to send audio.")

    sent_audio_index = 0

    while vc.is_connected():
        if len(sink.audio_data) > 0:
            audio_data = sink.audio_data[[*sink.audio_data.keys()][0]]
            if sent_audio_index < len(audio_data):    
                vc.send_audio_packet(audio_data[sent_audio_index], encode=True)
                sent_audio_index += 1

@bot.command()
async def end_echo(ctx: discord.ApplicationContext):
    await ctx.response.defer()
    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc: discord.VoiceClient = connections[ctx.guild.id][0]

        try:
            vc.stop_recording()
            await ctx.followup.send("Recording Ended.")
        except:
            await ctx.followup.send("Not recording.")
    else:
        await ctx.followup.send("Not in any voice channel!")

async def record_callback(sink: discord.sinks, channel: discord.TextChannel, *args):
    print("Echo ended.")


@bot.command()
async def llm_voice_channel(ctx: discord.ApplicationContext):
    await ctx.response.defer()
    if ctx.guild.id in connections:  # Check if the guild is in the cache.
        vc: discord.VoiceClient = connections[ctx.guild.id][0]

        sink = StreamSink()

        vc.start_recording(
            sink,  # The sink type to use.
            record_callback,  # What to do once done.
            ctx.channel  # The channel to disconnect from.
        )

        await ctx.followup.send("LLM voice chat started.")

        await ctx.guild.change_voice_state(channel=ctx.author.voice.channel, self_mute=True)

        result = asyncio.gather(*[llm(ctx, vc, sink)])
        # t = threading.Thread(target=llm, args=[ctx, vc, sink])
        # t.run()
    else:
        await ctx.followup.send("Not in any voice channel!")


def safe_slice(array, start, end):
    if start > len(array):
        return []
    if end > len(array):
        return array[start:len(array)]
    else:
        return array[start:end]


async def llm(ctx: discord.ApplicationContext, vc: discord.VoiceClient, sink: discord.sinks):
    messages = [
            {"role": "system", "content": "You are sales woman named Joanna. You work at Apple, and repsond to queries regarding latest Apple products and comparisons. Keep responses short and one sentence long."},
        ]   
    started = False
    print("Starting to record audio.")
    while vc.is_connected():
        if len(sink.audio_buffer) > 0:
            audio_data = sink.audio_buffer
            mono_audio_data = np.frombuffer(b"".join(audio_data), dtype=np.int16).ravel()[::2]
            if started:
                if is_end_of_speech(mono_audio_data):
                    print("Silence detected!")
                    stt_transcript = stt_transcribe(audio_data=mono_audio_data)
                    print(stt_transcript)
                    messages.append({"role": "user", "content": stt_transcript})
                    llm_response = llm_inference(messages=messages)
                    print(llm_response)
                    messages.append({"role": "assistant", "content": llm_response})
                    tts_audio_data = tts_generation(llm_response)
                    await ctx.guild.change_voice_state(channel=ctx.author.voice.channel, self_mute=False)
                    for i in range(0, len(tts_audio_data), 960*2):
                        vc.send_audio_packet(safe_slice(tts_audio_data, i, i+960*2).tobytes(), encode=True)
                        await asyncio.sleep(0.01)
                    await ctx.guild.change_voice_state(channel=ctx.author.voice.channel, self_mute=True)
                    sink.clear_audio_data()
                    started = False
            else:
                started = is_started(mono_audio_data)
                if started:
                        print("Speech started!")

def get_timestamps_from_int16(audio_data):
    combined = audio_data / (1 << 15)
    combined = combined.astype(np.float32)
    speech_timestamps = get_speech_timestamps(combined, vad_model, sampling_rate=48000)
    return speech_timestamps


def is_started(audio_data):
    if len(audio_data) > 0:
        speech_timestamps = get_timestamps_from_int16(audio_data)
        return len(speech_timestamps) > 0
    else:
        return False

def is_end_of_speech(audio_data):
    if len(audio_data) > 48000*2*2: # 2 seconds
        speech_timestamps = get_timestamps_from_int16(audio_data[-48000*2:])
        return not len(speech_timestamps) > 0
    else:
        return False
        


# async def once_done(sink: discord.sinks, channel: discord.TextChannel, *args):  # Our voice client already passes these in.
#     recorded_users = [  # A list of recorded users
#         f"<@{user_id}>"
#         for user_id, audio in sink.audio_data.items()
#     ]
#     await sink.vc.disconnect()  # Disconnect from the voice channel.
#     files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]  # List down the files.
#     await channel.send(f"finished recording audio for: {', '.join(recorded_users)}.")  # Send a message with the accumulated files.

# @bot.command()
# async def record(ctx):  # If you're using commands.Bot, this will also work.
#     voice = ctx.author.voice

#     if not voice:
#         await ctx.respond("You aren't in a voice channel!")

#     vc = await voice.channel.connect()  # Connect to the voice channel the author is in.
#     connections.update({ctx.guild.id: vc})  # Updating the cache with the guild and channel.

#     vc.start_recording(
#         sink,  # The sink type to use.
#         once_done,  # What to do once done.
#         ctx.channel  # The channel to disconnect from.
#     )
#     await ctx.respond("Started recording!")
    

# @bot.command()
# async def stop_recording(ctx):
#     if ctx.guild.id in connections:  # Check if the guild is in the cache.
#         vc = connections[ctx.guild.id]
#         vc.stop_recording()  # Stop recording, and call the callback (once_done).
#         del connections[ctx.guild.id]  # Remove the guild from the cache.
#         await ctx.delete()  # And delete.
#     else:
#         await ctx.respond("I am currently not recording here.")  # Respond with this if we aren't recording.

# ai = discord.SlashCommandGroup("ai", "AI commands")

# @ai.command()
# async def query_google(interaction: discord.Interaction, model: str, prompt: str):
#     google_llm = Google_LLM(model)
#     await interaction.response.defer()
#     await interaction.followup.send(google_llm.get_response(prompt=prompt))

# @ai.command()
# async def query_openai(interaction: discord.Interaction, model: str, prompt: str):
#     openai_llm = OpenAI_LLM(model)
#     await interaction.response.defer()
#     await interaction.followup.send(openai_llm.get_response(prompt=prompt))


# bot.add_application_command(ai)

if __name__ == "__main__":
    bot.run(os.getenv('DISCORD_BOT_TOKEN')) # run the bot with the token