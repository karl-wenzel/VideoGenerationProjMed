"""Central prompt configuration for image generation workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompositeGenerationOptions:
    """Concurrency and retry settings for the composite image pipeline."""

    character_reference_workers: int = 3
    layout_workers: int = 2
    scene_workers: int = 2
    pose_workers_per_scene: int = 3
    max_api_workers: int = 4
    max_retries: int = 3
    retry_base_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 20.0

USE_QWEN_COMPOSITE_SCENE_CHARACTER_MASK = True
QWEN_COMPOSITE_SCENE_CHARACTER_MASK_EROSION_PIXELS = 3

COMPOSITE_IMAGE_STYLE_PROMPT = (
    "Style: warm children's illustration, vivid colors, gentle natural light, "
    "soft painterly textures, brushstrokes, expressive, friendly subjects, "
    "coherent composition. Painted, happy style."
)

COMPOSITE_CHARACTER_LAYER_STYLE_PROMPT = (
    "Style: warm children's illustration, vivid colors, clean character "
    "silhouette, soft painterly textures, expressive but friendly character, "
    "coherent character design. Painted, happy style."
)

QWEN_COMPOSITE_SCENE_STYLE_PROMPT = (
    "Style: warm children's illustration, vivid colors, gentle natural light, "
    "soft painterly textures, detailed background, "
    "brushstrokes, coherent composition. Painted, happy style."
)

LLM_COMPOSITE_LAYOUT_SYSTEM_MESSAGE = (
    "You plan layered children's book image generation. Return only "
    "valid JSON. The JSON must contain background_prompt and "
    "characters. characters must be an array where each item has "
    "name, pose_prompt, placement_box, and layer_order. "
    "placement_box must contain x, y, width, and height as numbers "
    "from 0 to 1. Names are only for internal matching. "
    "background_prompt must describe only the setting. pose_prompt "
    "must describe a full-body isolated character pose that can be "
    "composited into the background. Infer concrete posture from "
    "the scene activity, including whether the body is standing, "
    "sitting, kneeling, crouching, reaching, looking, pointing, "
    "carrying, or resting. Include facing direction, gesture, "
    "expression, limb positions, and held objects. Keep "
    "pose_prompt limited to the isolated character body, clothing, "
    "held objects, expression, and action. Dont mention more than one character in a pose_prompt. Put all location, "
    "surface, scenery, lighting, weather, props that are not held, secondary characters "
    "and environment details in background_prompt instead. Use "
    "visual traits, dont mention proper names. Describe objectively and visually. Generated images "
    "must be suitable for children."
)

LLM_COMPOSITE_LAYOUT_USER_PREFIX = (
    "Create a composite layout plan for this scene. "
    "Use compact image prompts. Make each character box large "
    "enough for a clear full-body subject. The character pose "
    "prompts will be rendered separately on a solid backdrop, so "
    "they must be meaningful isolated body poses that visibly fit "
    "the scene action after compositing. Keep scene objects and "
    "setting details in background_prompt, and keep pose_prompt as "
    "a character-only layer prompt."
)

LLM_COMPOSITE_LAYOUT_REPAIR_USER_MESSAGE = (
    "Repair the previous response into only one valid JSON "
    "object with background_prompt and characters. Keep the "
    "same scene intent and preserve clear scene-matching body "
    "poses for every character."
)

COMPOSITE_CHARACTER_REFERENCE_PROMPT_LINES = (
    "Create a new character from scratch using the text description.",
    "Single full-body subject.",
    "Neutral standing pose.",
    "Centered composition.",
    "Uniform flat pure magenta #ff00ff studio backdrop from edge to edge.",
    "Clear silhouette and complete outfit.",
    "Consistent storybook illustration.",
    "No scenery, no ground patch, no floor contact surface, no cast shadow.",
)

QWEN_COMPOSITE_SCENE_REFERENCE_INSTRUCTION_LINES = (
    "The uploaded image is a character layout reference.",
    "Treat the visible uploaded characters as fixed foreground subjects.",
    "Use exactly those visible uploaded characters as the complete character cast.",
    "Only replace the flat pure magenta #ff00ff canvas with the requested environment.",
    "Keep the exact number of visible characters from the uploaded image.",
    (
        "Preserve their appearance, clothing, pose, scale, and relative "
        "positions from the uploaded image."
    ),
    (
        "Do not create, duplicate, mirror, move, resize, hide, or remove any "
        "visible character."
    ),
    (
        "Blend the environment around the existing character edges with "
        "natural contact shadows, local lighting, and ground or air contact."
    ),
    "Do not ignore the characters; compose the scene around their exact positions.",
    (
        "The final image must contain no extra copies, reflections, "
        "silhouettes, or background versions of the visible characters."
    ),
)

QWEN_COMPOSITE_SCENE_ACTION_SECTION_LABEL = "Scene action cue:"
QWEN_COMPOSITE_SCENE_ENVIRONMENT_SECTION_LABEL = "Environment to create:"

COMPOSITE_CHARACTER_IDENTITY_SUMMARY_PREFIX = "- Visible character:"

COMPOSITE_FALLBACK_BACKGROUND_PROMPT_LINES = (
    "Empty storybook setting.",
    "Open foreground space for later character compositing.",
)

COMPOSITE_POSE_CLARITY_PROMPT_LINES = (
    "Render a character-only studio layer.",
    "Show the scene activity through body posture, gesture, and expression.",
    "Keep the pose as an isolated subject layer for later compositing.",
    "Do not render scenery, landscape, ground, floor, grass, secondary characters, sky, weather, lighting effects, shadows, or environmental props.",
    "Only include objects physically held or worn by the character.",
)

COMPOSITE_DEFAULT_SCENE_DESCRIPTION = (
    "Gentle story moment in a picture book scene."
)

QWEN_COMPOSITE_CHARACTER_POSE_PROMPT_LINES = (
    "Use the provided reference image as the identity guide.",
    "Single full-body subject.",
    "Uniform flat pure magenta #ff00ff studio backdrop from edge to edge.",
    "Centered isolated subject.",
    "Clear silhouette.",
    "Character-only layer.",
    "No background objects, no scenery, no ground patch, no floor contact surface, no cast shadow.",
    "No environmental lighting or atmosphere; keep plain studio lighting.",
    "Scene-specific body pose and expression:",
)
