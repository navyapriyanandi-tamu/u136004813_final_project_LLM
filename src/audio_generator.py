"""
Audio Generator Module
Converts repository summaries into audio narrations with audio-optimized content
"""

from typing import Dict
from pathlib import Path
from gtts import gTTS
import anthropic
import config


class AudioGenerator:
    """Generates audio-optimized narrations from code summaries"""

    def __init__(self, repo_name: str):

        self.repo_name = repo_name
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model = config.LLM_MODEL
        self.audio_dir = Path("audio")
        self.audio_dir.mkdir(exist_ok=True)

    def create_audio_optimized_summary(self, human_summary: str) -> str:
        """
        Transform human summary into audio-optimized narration with analogies
        """
        prompt = f"""Transform this code repository summary into an engaging audio narration.

REPOSITORY: {self.repo_name}
ORIGINAL SUMMARY:
{human_summary}

AUDIO NARRATION REQUIREMENTS:
1. **Conversational tone**: Write like you're explaining to a friend over coffee
2. **Analogies**: Add 1-2 simple real-world analogies to explain technical concepts
3. **Structure**: Use clear transitions - "First...", "Next...", "Finally..."
4. **Accessibility**: Explain all acronyms (e.g., "CLI, which stands for command-line interface")
5. **Natural pauses**: Use "..." where listeners should digest information
6. **Hook**: Start with "Imagine..." or "Think of this as..." to engage listeners
7. **Length**: Keep under 2 minutes when spoken (~250-300 words)
8. **Storytelling**: Frame the code as solving a problem or serving a purpose

EXAMPLE TRANSFORMATION:
Before: "This is a Flask web application with user authentication."
After: "Think of this as a digital doorway to a web application. When visitors arrive, they first encounter a security checkpoint... the authentication system... which checks if they're allowed to enter. Once verified, Flask, the framework powering this app, guides them through different rooms... or web pages... showing them exactly what they're permitted to see."

OUTPUT ONLY THE AUDIO NARRATION TEXT (no metadata, no explanations):
"""

        print(f"   Generating audio-optimized summary")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.7,  
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        return response.content[0].text.strip()

    def text_to_speech(self, text: str, filename: str, lang: str = 'en') -> str:
        """
        Convert text to MP3 audio using gTTS
        """
        output_path = self.audio_dir / f"{filename}.mp3"

        print(f"   Converting to speech")
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(str(output_path))

        return str(output_path.absolute())

    def generate_repository_audio(self, human_summary: str) -> Dict[str, str]:
        """
        Generate audio narration for repository summary
        """
        # Generate audio-optimized text
        audio_text = self.create_audio_optimized_summary(human_summary)

        # Save script for reference
        script_file = self.audio_dir / f"{self.repo_name}_audio_script.txt"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(audio_text)

        # Convert to speech
        audio_file = self.text_to_speech(audio_text, f"{self.repo_name}_summary")

        # Calculate stats
        word_count = len(audio_text.split())
        estimated_duration = word_count / 150  # ~150 words per minute

        print(f"   Audio generated: {word_count} words (~{estimated_duration:.1f} min)")

        return {
            "audio_file": audio_file,
            "script_file": str(script_file.absolute()),
            "audio_text": audio_text,
            "word_count": word_count,
            "estimated_duration_minutes": round(estimated_duration, 1)
        }
