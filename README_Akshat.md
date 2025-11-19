# Realtime Voice Agent with Smart Interruption Handling

This branch documents my work on a human-like LiveKit Voice Agent capable of handling real-time conversations without reacting to every random “umm”, “hmm”, or background noise. The objective was to build a voice assistant that behaves like a patient listener — one that responds only when needed, and stops instantly when someone tries to interrupt.

Demo Video Link : [Click here](https://drive.google.com/drive/folders/1194v3IffxAeszm49iDyXslrtym5ZNz5y)
---

## Project Overview

I built a realtime conversational agent using:

- LiveKit Agents for the real-time voice pipeline
- Deepgram STT for transcribing speech
- ElevenLabs TTS for natural voice output
- OpenAI via LiveKit Inference as the LLM for responses
- A fully custom Interrupt Handler that makes the agent behave more like a real listener

The core idea: the agent shouldn’t panic or react to every hesitation; it should act as a calm and attentive listener.

---

## Key Additions & What I Built

### `interrupt_handler.py`

This is the main component I designed. It:

- Categorizes user speech into:
  - Ignore → fillers/noise
  - Interrupt → immediately stop the agent
  - Register → valid speech
- Detects multilingual fillers (English, Hindi, etc.) such as “umm”, “haan”, “acha”
- Allows runtime-configurable fillers through `.env`
- Tracks stats and provides detailed logs
- Handles fuzzy cases like “ummmmmm”, “hmmmmm”

### `realtime_voice_agent.py`

A customized LiveKit worker that integrates:

- Deepgram for STT
- ElevenLabs for TTS
- OpenAI (via LiveKit Inference)
- The new Interrupt Handler
- A conversational prompt designed for natural dialogue flow

### `test_interrupt_handler.py`

A focused test suite with more than 25 real scenarios, covering:

- low-confidence recognition
- mixed filler and meaningful speech
- Hindi + English stop phrases
- interruptions during agent speech
- fuzzy filler detection
- empty or partial text

---

## Current Capabilities

| Behavior                                   | Example Input                   | Result                                      |
| ------------------------------------------ | ------------------------------- | ------------------------------------------- |
| Ignores filler while agent is speaking     | “umm… hmm”                      | No interruption                             |
| Registers filler when agent is silent      | “umm, okay”                     | Treated as normal speech                    |
| Recognizes clear interruptions             | “wait”, “stop now”, “ruko zara” | Agent stops speaking                        |
| Interprets meaningful queries mid-speech   | “how does it work?”             | Current speech is stopped and answered      |
| Detects stretched filler sounds            | “ummmmmm”, “haaaaan”            | Correctly classified                        |
| Allows new filler words                    | via `.env`                      | Customisable at runtime                     |

---

## Challenges & Solutions

### 1. Filler vs Real Interruption
Users often mix them:
> “umm okay stop”

Solution: A scoring system based on filler ratio, priority words, STT confidence, and the agent’s speaking state.

### 2. Different Behavior Based on Who Is Speaking
When the agent is quiet, fillers often indicate thinking before speaking. This required a separate handling path depending on conversation state.

### 3. Multilingual Speech (Hinglish and Beyond)
People naturally switch languages. I used language dictionaries plus runtime extension support through `.env`.

### 4. STT Confidence Issues
Low-confidence noise was frequently misclassified. The handler now evaluates confidence levels to treat uncertain speech differently.

### 5. Natural Human Tone
I wrote a listening-focused conversational prompt that prevents robotic, filler-based responses.

---

## How to Test

### Run interrupt handler tests

python test_interrupt_handler.py

### Speak directly to the agent

python realtime_voice_agent.py console


Example interactions:

- Saying only ummm while the agent is speaking → nothing happens

- Saying stop please → agent immediately stops speaking

- Saying okay listen, can you help me → interruption + response

### Connect through LiveKit Playground / external client

python realtime_voice_agent.py start

### Environment Setup 

Create a .env file:

OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
ELEVEN_API_KEY=...
LIVEKIT_URL=wss://....livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...

Install dependencies:

pip install -r requirements.txt


### Conclusion

This branch upgrades a basic voice bot into a calm, respectful conversational agent. It:

- understands hesitation

- responds naturally

- interrupts intelligently

- supports multilingual speech

- adapts through configurable filler words

The focus wasn’t only on speaking, but on listening like a person.
The goal was not to generate more speech — but to understand when to remain silent.

