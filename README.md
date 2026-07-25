CLIPIndia 🇮🇳👗

Parameter-efficient contrastive fine-tuning of SigLIP for Indian fashion — with a fun text + image → image search app.

Off-the-shelf vision-language models like CLIP and SigLIP are trained mostly on Western imagery, so they stumble on Indian apparel: a kurta, a lehenga, a saree, or regional weaves often get mislabeled or retrieved poorly. CLIPIndia adapts Google's SigLIP to this domain using parameter-efficient fine-tuning (PEFT), then wraps it in a multimodal search app where you can query with text, an image, or both together.

✨ What it does
CLIPIndia supports three query modes, but the one that makes it fun is the third:
Text-to-image search — type "red banarasi silk saree" and get matching results
Image-to-image search — upload a photo and find visually similar apparel

🌟 Compositional search (image + text) — give it an image and a text tweak, and it retrieves items that keep the image's look but apply your edit
Example: upload a photo of a plain shirt, type "blue checks", and CLIPIndia returns shirts with the same silhouette but in a blue checked pattern. The image sets the base (it's a shirt, this cut, this style); the text steers the attributes (color, pattern) — so you're editing a visual query in natural language instead of starting over.

This works because both the image and the text land in the same SigLIP embedding space. CLIPIndia blends the two into a single query vector, then does nearest-neighbor search against the catalog — so "this shirt, but blue checked" becomes one combined point in that space rather than two separate searches.
Full fine-tuning of a VLM is expensive and tends to wreck general-domain performance. Instead, CLIPIndia uses PEFT to adapt only a small set of parameters via contrastive learning on a curated Indian clothing dataset. The goal: sharpen the model on regional apparel without forgetting everything else it knows.

Results:

+57.7% Recall@1 on regional apparel categories vs. the base model
< 2% degradation on general-domain retrieval — the adaptation stays targeted
Trained efficiently on a single GPU thanks to the parameter-efficient setup
