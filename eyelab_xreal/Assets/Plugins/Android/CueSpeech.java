package it.univpm.eyelab.probes;

import android.content.Context;
import android.media.AudioAttributes;
import android.speech.tts.TextToSpeech;

import java.util.Locale;

/**
 * T6.3: spoken prompts for the guided probe runs (Jack, 2026-11-18: "audio cues ... so that I can focus on what I
 * am seeing"). Android TextToSpeech, English. If no engine or no English voice is installed, {@link #state()} says
 * so and the Unity side falls back to tones only.
 */
public final class CueSpeech {
    private static TextToSpeech tts;
    private static volatile String status = "not started";
    private static volatile boolean ready;
    private static int utterance;

    private CueSpeech() { }

    /** Starts the engine; initialisation is asynchronous, poll {@link #state()}. */
    public static synchronized String init(Context ctx) {
        if (tts != null) return status;
        status = "initialising";
        try {
            tts = new TextToSpeech(ctx.getApplicationContext(), code -> {
                if (code != TextToSpeech.SUCCESS) { status = "init failed " + code; return; }
                int lang = tts.setLanguage(Locale.UK);
                if (lang < 0) lang = tts.setLanguage(Locale.US);
                if (lang < 0) { status = "no English voice (" + lang + ")"; return; }
                tts.setSpeechRate(1.05f);
                tts.setAudioAttributes(new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build());
                String engine = tts.getDefaultEngine();
                status = "ready, engine " + engine + ", voice " + (tts.getVoice() != null ? tts.getVoice().getName() : "?");
                ready = true;
            });
        } catch (Exception e) {
            status = "init exception " + e;
        }
        return status;
    }

    public static String state() { return status; }

    public static boolean isReady() { return ready; }

    /** Speaks at once, cutting off anything still being said. Returns false if not ready. */
    public static synchronized boolean say(String text) {
        if (!ready || tts == null) return false;
        return tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "eyelab" + (utterance++)) == TextToSpeech.SUCCESS;
    }

    public static synchronized void shutdown() {
        if (tts != null) { try { tts.stop(); tts.shutdown(); } catch (Exception ignored) { } }
        tts = null;
        ready = false;
        status = "shut down";
    }
}
