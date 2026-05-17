"""Central prompt configuration for image generation workflows."""

SCENE_IMAGE_STYLE_PROMPT = (
    "Style: warm children's illustration, vivid colors, gentle natural light, "
    "soft painterly textures, detailed background, brushstrokes, expressive but friendly characters, coherent "
    "composition. Painted, happy style."
)

COMPOSITE_IMAGE_STYLE_PROMPT = (
    "Style: warm children's illustration, vivid colors, gentle natural light, "
    "soft painterly textures, detailed background, brushstrokes, expressive but friendly characters, coherent "
    "composition. Painted, happy style."
)

LLM_SCENE_IMAGE_PROMPT_OUTPUT_CONTRACT = (
    "Return a compact image prompt with should include: "
    "character traits from the JSON, including clothing, "
    "hair, accessories, and any other visual identity fields; "
    "a brief description of the scene action and setting. "
)

LLM_SCENE_IMAGE_PROMPT_SYSTEM_MESSAGE = (
    "You write compact image prompts for image generation. "
    "Your main job is enforcing character consistency. Include "
    "all fixed visual character attributes provided by the user. "
    "Keep the scene description brief and avoid unnecessary "
    "background, composition, mood, prop, or lighting details. "
    "Do not choose or mention an art style, medium, rendering "
    "technique, camera model, artist, genre label, or visual "
    "finish. Return only the prompt text, with no "
    "markdown, labels, or extra explanation."
)

LLM_SCENE_IMAGE_PROMPT_USER_PREFIX = (
    "Use this JSON. Do not add style information."
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
    "held objects, expression, and action. Put all location, "
    "surface, scenery, lighting, weather, props that are not held, "
    "and environment details in background_prompt instead. Use "
    "visual traits, not proper names. Image prompts should use "
    "positive visual phrasing only."
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
    "Single full-body subject.",
    "Neutral standing pose.",
    "Centered composition.",
    "Plain solid light studio backdrop.",
    "Clear silhouette and complete outfit.",
    "Consistent storybook illustration.",
)

QWEN_COMPOSITE_SCENE_REFERENCE_INSTRUCTION_LINES = (
    "The uploaded image is a character layout reference.",
    "Use exactly the visible uploaded characters as the complete character cast.",
    (
        "Preserve their appearance, clothing, pose, scale, and relative "
        "positions from the uploaded image."
    ),
    (
        "Treat the flat neutral grey canvas as a replaceable layout field "
        "for the environment."
    ),
    (
        "Fill the layout field with the environment below, "
        "integrating the characters with natural ground contact, "
        "lighting, shadows, and depth."
    ),
)

QWEN_COMPOSITE_SCENE_CHARACTER_IDENTITY_SECTION_LABEL = "Character identity guide:"
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
    "Scene-specific body pose and expression:",
)
