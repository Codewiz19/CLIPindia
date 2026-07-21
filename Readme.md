CLIPIndia 🇮🇳👗

Parameter-efficient contrastive fine-tuning of SigLIP for Indian fashion — with a fun text + image → image search app.

Off-the-shelf vision-language models like CLIP and SigLIP are trained mostly on Western imagery, so they stumble on Indian apparel: a kurta, a lehenga, a saree, or regional weaves often get mislabeled or retrieved poorly. CLIPIndia adapts Google's SigLIP to this domain using parameter-efficient fine-tuning (PEFT), then wraps it in a multimodal search app where you can query with text, an image, or both together.

✨ What it does
Text-to-image search — "red silk banarasi saree" → matching results
Image-to-image search — upload a photo, find visually similar apparel
Combined text + image search — anchor on an image and steer with text ("this, but in blue")
🎯 Why PEFT + contrastive fine-tuning

Full fine-tuning of a VLM is expensive and tends to wreck general-domain performance. Instead, CLIPIndia uses PEFT to adapt only a small set of parameters via contrastive learning on a curated Indian clothing dataset. The goal: sharpen the model on regional apparel without forgetting everything else it knows.

Results:

+57.7% Recall@1 on regional apparel categories vs. the base model
< 2% degradation on general-domain retrieval — the adaptation stays targeted
Trained efficiently on a single GPU thanks to the parameter-efficient setup
