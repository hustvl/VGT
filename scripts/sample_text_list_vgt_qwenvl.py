import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
from xtuner.registry import BUILDER
from mmengine.config import Config
from einops import rearrange
import sys

from src.utils import load_checkpoint_with_ema


def _generate_text_to_image(model, prompts, cfg_scale=3.5, num_steps=50,
                           height=512, width=512, temperature=1.0, grid_size=2, **kwargs):
    """
    Text-to-image generation task (prompts can be a single string or a list)
    
    Args:
        return_list: bool, whether to return a list of images
    
    Returns:
        PIL.Image or list[PIL.Image]
    """
    if not isinstance(prompts, list):
        prompts = [prompts]

    bsz = grid_size ** 2
    
    # Use new batch text condition preparation function
    batch_text_conditions = model.prepare_batch_text_conditions(prompts)
    input_ids = batch_text_conditions['input_ids']
    attention_mask = batch_text_conditions['attention_mask']
    
    # input_ids and attention_mask already contain the mixture of prompt and CFG
    # First half is prompt, second half is CFG
    total_prompts = len(prompts)
    seq_len = input_ids.shape[1]
    
    # Reshape to [total_prompts, 2, seq_len] format, where 2 represents prompt+CFG
    assert 2*total_prompts == input_ids.shape[0], "prepare_batch_text_conditions will return input_ids containing cfg. [p1,p2,p3,cfg1,cfg2,cfg3]"
    
    # Expand batches
    prompt_input_ids = input_ids[:total_prompts].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    prompt_attention_mask = attention_mask[:total_prompts].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    cfg_input_ids = input_ids[total_prompts:].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    cfg_attention_mask = attention_mask[total_prompts:].unsqueeze(1).expand(-1, bsz, -1).flatten(0, 1)
    
    # Concatenate into [prompt1, prompt2, ..., cfg1, cfg2, ...] format
    batch_input_ids = torch.cat([prompt_input_ids, cfg_input_ids], dim=0)
    batch_attention_mask = torch.cat([prompt_attention_mask, cfg_attention_mask], dim=0)

    if cfg_scale == 1.0:
        batch_input_ids = batch_input_ids[:bsz]
        batch_attention_mask = batch_attention_mask[:bsz]

    # Generate images
    samples = model.generate(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
        cfg_scale=cfg_scale,
        num_steps=num_steps,
        height=height,
        width=width,
        temperature=temperature,
        num_image_pre_caption=bsz,
        **kwargs
    )
    
    gen_images = []
    # Split results and process images corresponding to each prompt
    for i, prompt in enumerate(prompts):
        # Each prompt corresponds to bsz images, directly take corresponding positions
        start_idx = i * bsz
        end_idx = start_idx + bsz
        prompt_samples = samples[start_idx:end_idx]
        
        prompt_samples = rearrange(prompt_samples, '(m n) c h w -> (m h) (n w) c', m=grid_size, n=grid_size)
        prompt_samples = torch.clamp(
            127.5 * prompt_samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        gen_images.append(Image.fromarray(prompt_samples))

    
    return gen_images



def load_model(CONFIG_PATH, model_path):
    """Load model and weights"""
    # Load configuration and model
    config = Config.fromfile(CONFIG_PATH)
    model = BUILDER.build(config.model).cuda().bfloat16().eval()

    print("Model loading completed!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model precision: {next(model.parameters()).dtype}")

    # Load checkpoint
    if model_path is not None:
        load_checkpoint_with_ema(model, model_path, use_ema=True, map_location='cpu', strict=False)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"Weights loading completed: {model_path}")
    else:
        print("No checkpoint path specified")

    return model


def main():
    # Parameter settings
    ###############VGT Qwen2.5VL########################
    CPKT_PATH = "ckpts/hustvl/vgt_qwen25vl_2B_sft/iter_5000.pth"
    CONFIG = "configs/models/vgt_qwen25vl_2B_448px.py"

    ###############VGT InterVL3########################
    # CPKT_PATH = "ckpts/hustvl/vgt_internvl3_1_6B_sft/iter_5000.pth"
    # CONFIG = "configs/VGT_internvl3/vgt_internvl3_1_6B_448px.py"
    
    # default
    CFG_SCALE = 4.5
    NUM_STEPS = 100
    HEIGHT = 448
    WIDTH = 448
    GRID_SIZE = 2 # per prompts sample GRID_SIZE*GRID_SIZE
    scheduler_type = "random"
    BTACH_SIZE = 8
    acc_ratios = [1,2,8,16] # 1(next token), 4, 16, 32, ..., 256

    MODEL_NAME = os.path.splitext(os.path.basename(CONFIG))[0]
    OUTPUT_DIR = f"./save_fig/{MODEL_NAME}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Load model
    model = load_model(CONFIG, model_path=CPKT_PATH)
    
    print("🔄 Preparing to generate images...")
    
    prompts = [
        # ==================== Human ====================
        "A young East Asian girl with braided hair and a floral shirt poses against a plain white background, looking directly at the camera with a slight smile.",
        "A young woman with her hair in a bun sits in a boat, wearing a striped off-the-shoulder dress, with a blurred background of trees and water.",
        "A fair-skinned woman with blonde hair and a white orchid in her hair is looking at the camera with a slight smile against a white background.",
        "A close-up shot features a young woman with long, ombre hair and blue eyes, wearing a dark gray shirt against a muted gray background.",
        
        "An elderly man with silver hair and weathered skin sits in warm window light, deep wrinkles telling stories of age, kind blue eyes gazing thoughtfully into the distance.",
        "A young boy with freckles and tousled red hair laughs joyfully, golden hour sunlight streaming from the side creating a warm glow on his face and shoulders.",
        "A woman with dark curly hair wears a flowing emerald green dress, standing against a textured cream wall, natural light creating soft shadows across her features.",
        "A man in a white shirt sits reading by a window, morning light illuminating the side of his face, creating a peaceful and contemplative mood with soft blue tones.",
        "A teenage girl with long black hair and bangs looks over her shoulder, wearing a denim jacket, soft pink sunset light creating a dreamy atmospheric glow.",
        "A woman with a sleek bob haircut and red lipstick poses against a black background, dramatic side lighting creating strong contrast and highlighting her defined cheekbones.",
        "A child with blonde curls holds a dandelion, soft afternoon light filtering through the fluffy seeds, wonder and curiosity evident in their wide blue eyes.",
        "A bearded man in his thirties wears a gray turtleneck sweater, standing against a minimalist white background, soft even lighting creating a calm professional portrait.",
        "A woman with auburn hair in loose waves wears a cream knit sweater, sitting by a rain-streaked window, melancholic expression as she gazes outside at the gray day.",
        "A young woman with braided crown hairstyle and delicate gold jewelry looks directly at the camera, warm brown skin glowing in soft diffused studio light.",
        "A man in a black leather jacket stands in misty morning fog, droplets of moisture on his short dark hair, dramatic moody lighting with cool blue tones.",
        "A girl with space buns hairstyle and freckles smiles brightly, wearing a yellow sundress, warm golden light creating a cheerful and energetic atmosphere.",
        "A woman with silver-gray hair in an elegant updo wears a black silk blouse, photographed against deep burgundy velvet, sophisticated studio lighting emphasizing textures.",
        "A young man with wet hair pushed back and water droplets on his skin looks intensely at the camera, dramatic lighting creating strong shadows on his jawline.",
        "A woman wearing a white hijab stands against a soft pink background, gentle lighting illuminating her peaceful expression and the delicate fabric draping.",
        "A boy in a striped shirt sits cross-legged with paint on his hands and face, colorful backdrop blurred, natural light highlighting his creative mess and joyful expression.",
        "A woman with long straight blonde hair wears a light blue chambray shirt, photographed in soft overcast light giving even, flattering illumination to her features.",
        "A dancer in mid-movement wears flowing white fabric that catches the air, dramatic backlighting creating a silhouette effect with rim light outlining her form.",
        "A man with round glasses and a gentle smile sits in a cozy library corner, warm lamp light creating an intimate scholarly atmosphere with books softly blurred behind.",
        "A woman with dark braided hair adorned with small flowers looks serenely to the side, warm sunset light creating a golden glow on her skin and the petals.",
        "A young professional in a crisp white button-down shirt poses against a minimal gray background, clean studio lighting creating a modern corporate portrait.",
        "A woman with short platinum blonde hair and bold makeup looks confidently at the camera, cool-toned lighting against a light blue background creating an editorial fashion feel.",
        "A man in traditional white linen clothing stands in natural daylight, fabric texture and weave clearly visible, serene expression against a soft beige wall.",

        # ==================== Animal ====================
        "A close-up shot of a young white snow fox with alert ears and intelligent eyes, standing amid soft falling snow.",
        "The image features a close-up of a dog with a brown coat and a white patch on its chest, looking to the left against a warm brown background.",
        "A close-up portrait of a red panda, its fur a mix of rich reddish-brown and white, with distinctive black markings around its eyes.",
        
        "A majestic Bengal tiger rests on a moss-covered rock, its orange and black striped fur gleaming in dappled forest sunlight filtering through leaves above.",
        "A snowy owl perched on a frozen branch, its pure white feathers flecked with black spots, piercing yellow eyes gazing forward against a soft gray winter sky.",
        "A golden retriever puppy sits in tall grass during golden hour, its fluffy cream-colored fur backlit by warm sunset rays, tongue out in a playful expression.",
        "A black panther emerges from dark shadows, its sleek coat reflecting subtle blue highlights, amber eyes glowing intensely against a blurred jungle background.",
        "A hummingbird hovers mid-air with iridescent green and purple feathers catching the light, its delicate wings frozen in motion against a soft bokeh garden backdrop.",
        "A gray tabby cat sits by a sunlit window, dust particles floating in the warm beam of light that illuminates its striped fur and bright green eyes.",
        "A emperor penguin chick stands on Antarctic ice, its fluffy gray down feathers ruffling in the cold wind, with soft blue ice formations behind.",
        "A monarch butterfly rests on a purple wildflower, its orange and black wings spread wide showing intricate patterns, morning dew visible on delicate wing scales.",
        "A white Siberian husky lies in fresh snow, pale blue eyes looking directly at the camera, individual snowflakes caught in its thick double coat.",
        "A dolphin breaches the ocean surface, water droplets frozen mid-air around its sleek gray body, late afternoon sun creating golden highlights on wet skin.",
        "A koala clings to a eucalyptus tree trunk, its fluffy gray fur and large round nose in sharp focus, with soft green foliage blurred in the background.",
        "A peacock displays its magnificent tail feathers in full spread, the iridescent blues and greens catching sunlight against a neutral stone courtyard.",
        "A sea turtle glides through crystal clear blue water, sunlight rays penetrating the surface creating dramatic light beams around its ancient shell.",
        "A fox squirrel sits upright holding an acorn, its bushy orange tail curled behind, fur backlit creating a glowing rim light against autumn bokeh.",
        "A blue morpho butterfly perched on a green leaf, its metallic blue wings partially open revealing the stunning electric blue coloration in soft rainforest light.",
        "A arctic hare sits alert in snow, its pure white winter coat blending with surroundings, only its black-tipped ears and dark eyes creating contrast.",
        "A koi fish swims near the pond surface, its orange, white and black patterns clearly visible through clear water, lily pads creating soft shadows above.",
        "A lynx sits on a snowy hillside, tufted ears prominent, thick spotted gray fur detailed in soft overcast light, pale yellow eyes focused intently forward.",
        "A chameleon grips a thin branch, its scaly skin displaying vivid green and blue colors, one eye focused forward while the other looks sideways independently.",
        "A seal pup rests on white ice, its spotted gray coat still fluffy, large dark eyes and whiskers prominent against the pristine arctic landscape.",
        "A toucan perches on a tropical branch, its enormous colorful beak showing yellow, orange and green bands, black plumage contrasting sharply with bright foliage behind.",
        "A fennec fox sits in warm sand, its enormous ears backlit by desert sunset, cream-colored fur glowing golden, dark intelligent eyes looking curiously at the camera.",


        # ==================== Landscape  ====================
        "The image depicts a stunning mountainous landscape during sunset, featuring snow-covered peaks and rugged terrain against a clear sky transitioning from blue to orange.",
        "The image captures a scenic view of a mountainous landscape during sunrise, with a person silhouetted on a rocky cliff against bright morning light.",
        
        "A lone tree stands on a rolling green hill, its branches silhouetted against a dramatic orange and purple sunset sky with wispy clouds.",
        "A wooden dock extends into a perfectly still mountain lake, its reflection creating a mirror image in the glassy turquoise water, snow-capped peaks in the distance.",
        "A winding dirt path cuts through a field of purple lavender, leading toward distant blue hills under a clear summer sky with a single white cloud.",
        "A lighthouse stands on a rocky coastal cliff at twilight, its beam cutting through the misty blue hour atmosphere, waves crashing on rocks below.",
        "A bamboo forest path is illuminated by soft filtered sunlight streaming through the dense green canopy, creating dramatic rays of light in the morning mist.",
        "A sand dune ripples across the frame in warm golden light, long shadows emphasizing its curves and texture against a gradient sky from orange to deep blue.",
        "A frozen waterfall clings to dark rock face, icicles catching the pale winter sunlight, surrounded by snow-covered evergreen branches.",
        "A single red barn sits in a snow-covered field, warm light glowing from its windows against the cold blue twilight sky, bare trees framing the scene.",
        "A narrow canyon with smooth curved sandstone walls glows in shades of orange and red, a beam of sunlight illuminating the striated rock formations.",
        "A field of golden wheat sways gently, individual stalks catching the warm afternoon light, with a soft blue sky and distant tree line on the horizon.",
        "A misty forest clearing reveals a carpet of green moss on ancient rocks, soft diffused light filtering through fog, creating an ethereal atmosphere.",
        "A volcanic black sand beach meets turquoise ocean waves, smooth wet stones reflecting the overcast sky, a lone sea stack rising from the water.",
        "A autumn maple tree stands alone with brilliant red and orange foliage against a clear blue sky, fallen leaves scattered on the grass below.",
        "A desert rock arch frames distant layered mesas, warm evening light painting the red sandstone in rich tones against a fading turquoise sky.",
        "A crystal-clear stream flows over smooth river rocks, water creating white ripples around stones, surrounded by lush green ferns in soft forest light.",
        "A prairie landscape stretches to the horizon under dramatic storm clouds, a single ray of sunlight breaking through to illuminate the golden grassland.",
        "A cherry blossom tree in full bloom stands beside a traditional stone lantern, pink petals floating in the air against a soft gradient sky.",
        "A fjord reflects steep mountain cliffs in its perfectly calm dark blue water, early morning mist clinging to the peaks, creating a serene Nordic scene.",
        "A cactus stands prominently in the foreground against a desert sunset, its spines backlit creating a glowing outline, purple mountains silhouetted in the distance.",
        "A winding river cuts through a valley of autumn colors, aerial view showing the meandering water reflecting the cloudy sky, surrounded by red and gold foliage.",
        "A solitary lighthouse keeper's cottage sits on a green grassy island, surrounded by rough gray ocean water under moody storm clouds with dramatic lighting.",
        "A rice terrace cascades down the hillside in geometric patterns, flooded fields reflecting the blue sky and white clouds like mirrors, green seedlings visible underwater.",
        "A redwood tree trunk dominates the frame, its massive striated bark texture detailed in soft forest light filtering from above, ferns growing at its base.",

        # ==================== Art and Illustration Style ====================
        "The image features a decorative arrangement of flowers and leaves in a black vase, with large pink and yellow flowers against a light blue background.",
        "A striking piece of art features a woman's face split into two halves - one realistic, one abstract and colorful - adorned with a floral crown against dark background.",

        "A watercolor illustration of a single peony flower in full bloom, soft pink petals with delicate gradients bleeding into white, loose brushstrokes on textured paper.",
        "An art nouveau style portrait of a woman with flowing hair intertwined with decorative golden vines and flowers, elegant curved lines against a mint green background.",
        "A minimalist ink drawing of a crane standing on one leg, simple black brushstrokes capturing the bird's elegant form against blank white space.",
        "A stained glass window design depicting a hummingbird surrounded by geometric floral patterns, vibrant jewel tones of blue, green, and amber with black leading.",
        "An art deco style poster featuring a streamlined luxury train, bold geometric shapes in gold, black and cream, strong diagonal composition with stylized clouds.",
        "A Japanese ukiyo-e inspired illustration of a koi fish swimming upward through stylized waves, rich blues and oranges with fine linear details and patterns.",
        "A chalk pastel drawing of a ballet shoe on its ribbon, soft blended pinks and purples creating a dreamy atmospheric effect on dark gray paper.",
        "A papercut art silhouette of a deer in a forest, intricate layered black paper creating depth, backlit to show delicate cut details of foliage and antlers.",
        "A vintage botanical illustration of an iris flower, precise line work colored with subtle watercolors, scientific labels in elegant script on aged cream paper.",
        "An abstract acrylic painting of ocean waves, thick impasto texture in layers of turquoise, white and deep blue, capturing movement with bold palette knife strokes.",
        "A gouache painting of a cottage in a wildflower meadow, soft matte colors in pastel blues, purples and pinks, flat stylized shapes with dreamy fairytale quality.",
        "A linocut print of a lighthouse, bold black carved lines creating strong contrast with white negative space, simplified geometric forms with visible printing texture.",
        "An isometric pixel art illustration of a cozy reading nook, chair and bookshelf rendered in limited color palette, clean lines and charming retro game aesthetic.",
        "A colored pencil drawing of a glass perfume bottle, hyperrealistic rendering showing reflections and transparency, soft gradients on smooth white paper.",
        "A folk art style painting of a rooster, flat bright colors in red, yellow and green, decorative patterns on feathers, naive and cheerful composition.",
        "An oil painting of a single red rose in a crystal vase, classical still life with dramatic chiaroscuro lighting, rich colors and visible brushstrokes.",
        "A vector illustration of a coffee cup with steam, clean geometric shapes in warm browns and creams, modern flat design with subtle gradients.",
        "A scratchboard art piece of an owl, white lines revealed on black surface creating feather texture and piercing eyes, high contrast dramatic effect.",
        "A digital painting of a fantasy crystal, translucent geometric form catching light, soft glows and refractions in purples and blues against dark background.",
        "A traditional Chinese brush painting of bamboo stalks and leaves, expressive black ink with varying tones and flowing calligraphic quality on rice paper.",
        "A marker illustration of a vintage typewriter, bold outlines with vibrant teal and orange coloring, retro aesthetic with halftone dot patterns for shading.",
        "A mixed media collage featuring a bird's nest with blue eggs, combining watercolor, pressed flowers, and gold leaf on textured handmade paper.",
        "A pointillism style artwork of a sunflower, thousands of small dots in yellows, oranges and greens creating form through color and density variation.",

        # ==================== Objects ====================
        "A vintage brass telescope sits on a wooden tripod, its polished metal reflecting warm library light, leather accents and intricate focusing knobs in sharp detail.",
        "A white porcelain teacup with delicate blue floral pattern sits on a matching saucer, steam rising from hot tea, soft window light creating gentle shadows.",
        "A red electric guitar leans against a textured gray wall, glossy finish reflecting studio lights, chrome hardware and worn fretboard showing character and use.",
        "A leather-bound journal lies open with a fountain pen across its pages, warm desk lamp illuminating cream paper and handwritten text, brass pen clip catching light.",
        "A crystal chandelier hangs against a soft neutral background, individual prisms catching light and creating rainbow refractions, elegant metal framework visible.",
        "A vintage film camera with worn black leather and chrome details rests on dark wood surface, lens reflecting soft light, mechanical dials showing decades of use.",
        "A bonsai tree in a shallow ceramic pot sits centered on white cloth, twisted trunk and delicate green foliage in sharp focus, minimal Japanese aesthetic.",
        "A antique pocket watch with ornate gold engraving lies open, revealing intricate clockwork mechanism, chain coiled beside it on rich burgundy velvet.",
        "A single origami crane folded from red paper casts a sharp shadow on white background, precise geometric folds catching directional light to show dimension.",
        "A glass terrarium sphere contains small succulents and moss, suspended by copper wire against soft gray backdrop, condensation droplets on interior surface.",
        "A weathered brass compass rests on an old map, needle pointing north, patina and scratches showing age, soft museum-quality lighting emphasizing texture.",
        "A handmade ceramic vase with crackle glaze in celadon green sits empty against cream background, irregular shape and dripping glaze creating organic beauty.",
        "A vintage typewriter with cream keys and black body sits ready for use, ribbon visible, carriage return lever catching sidelight, evoking nostalgic writing atmosphere.",
        "A crystal perfume bottle with cut glass stopper reflects prismatic colors, art deco design clear against soft white silk fabric, elegant and luxurious presentation.",
        "A traditional Japanese tea whisk made of bamboo stands upright casting distinct shadow, fine tines splayed in perfect circle, soft natural lighting on neutral background.",
        "A silver pocket knife with mother-of-pearl handle lies partially open on dark stone, blade reflecting light, craftsmanship details visible in close-up.",
        "A hourglass with dark wood frame and white sand shows time flowing, top chamber nearly empty, dramatic lighting creating strong shadows on weathered desktop.",
        "A calligraphy brush with bamboo handle rests in a simple stone holder, black bristles in perfect point, ink drop suspended on tip catching light.",
        "A jeweled crown sits on burgundy velvet cushion, gold filigree work and colored gemstones catching multiple light sources, regal and detailed presentation.",
        "A antique skeleton key with ornate bow design lies on aged parchment paper, iron patina showing rust and history, warm candlelight creating atmosphere.",
        "A pair of worn leather work boots sits side by side, laces loosened, scuffs and creases telling stories, soft natural light emphasizing texture and character.",
        "A glass snow globe contains a miniature winter village scene, base details visible, slight reflection on surface, held in moment of settled snow.",

        # ==================== Fantasy & Magical Elements ====================
        "A single glowing blue mushroom emerges from dark forest floor, bioluminescent cap illuminating surrounding moss, ethereal magical atmosphere with soft particle effects.",
        "A crystal ball sits on an ornate bronze stand, swirling mists of purple and blue visible inside the sphere, mystical energy seeming to emanate from within.",
        "A fairy wing lies delicate and translucent on a leaf, iridescent membrane catching light like stained glass, intricate vein patterns visible in magical glow.",
        "A floating island fragment hovers in misty air, green grass and flowers growing on top, exposed rock and roots underneath, soft clouds drifting past.",
        "A dragon egg rests in a nest of silver fabric, deep emerald scales reflecting candlelight, subtle inner glow suggesting life within, mystical atmosphere.",
        "A magical staff made of twisted wood tops with glowing crystal, runes carved along length emanating soft blue light, standing upright against stone wall.",
        "A phoenix feather burns with gentle flame that doesn't consume it, orange and gold fire-light transitioning to red at edges, floating against dark background.",
        "A unicorn horn spirals elegantly with pearl-like iridescence, mounted on velvet display, catching light to show rainbow shimmer, mythical artifact presentation.",
        "A potion bottle contains swirling luminous liquid in shades of violet and teal, cork stopper sealed with wax, sitting on ancient spellbook with arcane symbols.",
        "A magical mirror frame of silver vines and leaves shows not a reflection but swirling galaxy of stars, cosmic portal effect in ornate border.",
        "A witch's hat in deep purple velvet with wide brim sits perfectly formed, silver moon and star embroidery catching light, buckle gleaming on ribbon.",
        "A enchanted rose under glass dome glows faintly red, petals perfect and suspended in time, dark wood base with brass plaque, beauty and beast aesthetic.",
        "A mermaid scale the size of a palm shimmers with every color, holding it shows shifting rainbow iridescence like abalone shell, magical oceanic artifact.",
        "A spell scroll partially unrolled reveals glowing runes and symbols, parchment edges slightly singed, floating ink particles creating mystical effect.",
        "A fairy door in miniature stands against tree bark, ornate carved wood with tiny handle, warm inviting glow emanating from crack beneath door.",
        "A wizard's orb levitates above an open palm (hand only, no full person), sphere of crackling purple energy, lightning arcs dancing across surface.",
        "A frozen snowflake impossibly large and detailed hovers in air, each crystalline branch perfect and unique, backlit with cool blue magical glow.",
        "A key made entirely of ice doesn't melt, intricate baroque design with frost patterns, suspended in beam of moonlight against velvet midnight blue.",
        "A dragon scale the size of a shield shows metallic sheen, overlapping ridges in deep garnet red, battle-worn edges, legendary treasure presentation.",
        "A magical hourglass contains not sand but flowing starlight, silver frame with celestial engravings, glowing particles defying gravity in mesmerizing patterns.",
        "A enchanted quill writes by itself on parchment, golden feather glowing softly, ink flowing to form elegant script, surrounded by soft magical sparkles.",
        "A moonstone pendant hangs on silver chain, gem glowing with inner light, surrounded by fine mist, rotating slowly to show blue-white adularescence.",

        # ==================== Vehicles & Transportation ====================
        "A vintage red bicycle leans against a brick wall, chrome handlebars catching afternoon sun, leather saddle worn with character, flowers in wicker basket.",
        "A classic convertible sports car in British racing green, polished chrome details and wire wheels gleaming, photographed at three-quarter angle against minimal gray background.",
        "A hot air balloon floats in clear blue sky, vibrant red and yellow striped envelope fully inflated, wicker basket visible below, soft morning light.",
        "A steam locomotive's front engine emerges from white steam clouds, black boiler with brass fittings and large round headlamp, powerful industrial beauty.",
        "A wooden sailboat with white canvas sails glides on calm turquoise water, rigging and ropes detailed, hull reflecting in glassy surface, serene nautical scene.",
        "A vintage motorcycle with chrome engine and black leather seat sits on display, fuel tank polished to mirror finish, classic mechanical details emphasized.",
        "A yellow taxi cab parked on wet pavement reflects city lights, rain droplets on hood and windshield, iconic checkered stripe visible, moody urban atmosphere.",
        "A space shuttle launches with massive plume of white smoke and orange flame, powerful vertical ascent against bright blue sky, historic moment captured.",
        "A gondola floats on Venetian canal, ornate black boat with brass fixtures and striped mooring pole, reflections in green water, romantic Italian atmosphere.",
        "A vintage propeller airplane with silver metal fuselage sits on tarmac, three-blade propeller and radial engine detailed, aviation golden age aesthetic.",
        "A double-decker red bus drives past in slight motion blur, iconic London transport against neutral background, vintage charm and classic design.",
        "A cable car suspended on wires climbs steep hill, passengers visible through windows, mechanical grip system shown, San Francisco style transportation.",
        "A speedboat cuts across blue water creating white wake spray, sleek fiberglass hull in metallic blue, chrome windshield and powerful outboard motors.",
        "A vintage horse-drawn carriage with burgundy leather seats and brass lanterns sits ready, large wooden wheels and harness details, classical elegance.",
        "A modern bullet train with aerodynamic white nose passes at high speed, slight motion blur showing velocity, sleek futuristic Japanese design.",
        "A classic Vespa scooter in mint green, chrome mirrors and headlight, parked on cobblestone street, Italian la dolce vita lifestyle embodied.",
        "A fire truck in bright red with extended ladder, chrome bells and hoses, emergency vehicle details emphasized, heroic service vehicle presentation.",
        "A rickshaw with colorful painted decorations sits empty, traditional Southeast Asian transport, intricate details on canopy and wheels, cultural transportation art.",
        "A vintage tram car with wooden interior visible through windows, brass bell and trolley pole, classic city transportation on metal tracks, nostalgic scene.",
        "A seaplane rests on calm lake water, floats reflected in mirror surface, propeller and high wing configuration, adventure aviation in natural setting.",
        "A skateboard deck shows colorful graphic art on underside, worn grip tape on top, trucks and wheels detailed, urban sport culture and street style.",
        "A classic pickup truck in weathered turquoise paint, chrome bumper and round headlights, farm vehicle character and American vintage automobile.",
    ]

    total = len(prompts)
    num_batches = (total + BTACH_SIZE - 1) // BTACH_SIZE
    print(f"🚀 Total prompts: {total}, batch size = {BTACH_SIZE}, total {num_batches} batches")

    global_index = 0  # Global index for saving filenames

    for b in range(num_batches):
        start = b * BTACH_SIZE
        end = min((b + 1) * BTACH_SIZE, total)
        batch_prompts = prompts[start:end]

        for acc_ratio in acc_ratios:
            print(f"\n===== 🟦 Acc_ratio:{acc_ratio} | Batch {b+1}/{num_batches}: prompts[{start}:{end}] =====")

            # ======== Generate images ========
            samples = _generate_text_to_image(
                model=model,
                prompts=batch_prompts,
                cfg_scale=CFG_SCALE,
                num_steps=NUM_STEPS,
                height=HEIGHT,
                width=WIDTH,
                seed=42,
                grid_size=GRID_SIZE,
                scheduler_type=scheduler_type,
                acc_ratio=acc_ratio,
            )

            # ======== Save images ========
            print("💾 Saving images...")
            for sample, batch_index in zip(samples, range(start,end)):
                try:
                    # Convert to PIL Image
                    if isinstance(sample, np.ndarray):
                        if sample.max() <= 1.0:
                            sample = (sample * 255).astype(np.uint8)
                        else:
                            sample = sample.astype(np.uint8)
                        img = Image.fromarray(sample)
                    else:
                        img = sample

                    # Save path (global incremental index)
                    filename = f"sample_{batch_index:06d}_acc_ratio_{acc_ratio}x.png"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    img.save(save_path, quality=95)

                except Exception as e:
                    print(f"❌ Failed to save sample {batch_index}: {e}")

                global_index += 1

        print(f"✔ Batch {b+1} saving completed (total {global_index} images)")

    print(f"\n🎉 All image generation completed, saved {global_index} images in total")
    print(f"📁 Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()