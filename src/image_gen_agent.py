import requests
import urllib.parse
import json
import time

class OceanImageGenerator:
    """
    Generates tactical ocean imagery using:
    1. Pollinations.ai (Primary: Free, Unlimited, No Key)
    2. Leonardo.ai (Fallback: High Quality, Requires Key)
    """

    def __init__(self, leonardo_key=None):
        self.leo_key = leonardo_key
        self.leo_url = "https://cloud.leonardo.ai/api/rest/v1/generations"

    def generate_scenario(self, location, risk_level, chemical_data, model="flux", tactical_context=None):
        """
        Orchestrates the generation process. Returns dict with 'url', 'source', 'prompt'.
        """
        # 1. Construct a detailed tactical prompt
        prompt = self._construct_prompt(location, risk_level, chemical_data, tactical_context)
        
        # 2. Generate Pollinations URL with selected model
        target_model = model
        final_prompt = prompt
        
        if model == "flux-realism":
            target_model = "flux"
            final_prompt = f"{prompt}, raw photo, hyper-realistic, 8k, highly detailed"

        if model == "flux-3d":
            target_model = "flux"
            final_prompt = f"{prompt}, 3D render, unreal engine 5, volumetric lighting, isometric view, octane render"

        # Sanitize prompt to avoid URL URL issues with Pollinations
        safe_prompt = final_prompt.replace("?", "").replace("&", "").replace("#", "")
        encoded_prompt = urllib.parse.quote(safe_prompt)
        
        pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&model={target_model}&nologo=true&seed={int(time.time())}"
        
        return {
            "url": pollinations_url,
            "source": f"Pollinations.ai ({model.upper()})",
            "prompt": prompt,
            "provider": "pollinations"
        }

    def generate_with_leonardo(self, location, risk_level, chemical_data, tactical_context=None):
        """
        Explicitly calls Leonardo AI if the user forces it or fallbacks are triggered.
        """
        if not self.leo_key:
            return {"error": "No Leonardo API Key provided."}

        prompt = self._construct_prompt(location, risk_level, chemical_data, tactical_context)
        
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.leo_key}"
        }
        
        # DEBUG LOGGING (Remove in production)
        print(f"DEBUG: Headers Auth: {headers['authorization'][:15]}...")
        print(f"DEBUG: Payload Prompt: {prompt[:50]}...")
        
        payload = {
            "height": 576,
            "width": 1024,
            "prompt": prompt,
            "modelId": "e316348f-7773-490e-adcd-46757c738eb7", # Absolute Reality v1.6
            "num_images": 1
        }
        
        try:
            # 1. Init Generation
            response = requests.post(self.leo_url, json=payload, headers=headers)
            if response.status_code != 200:
                return {"error": f"Leonardo Init Failed: {response.text}"}
                
            generation_id = response.json()['sdGenerationJob']['generationId']
            
            # 2. Poll for result
            for _ in range(10): # Wait up to 10 seconds
                time.sleep(2)
                check_url = f"{self.leo_url}/{generation_id}"
                status_resp = requests.get(check_url, headers=headers)
                status_data = status_resp.json()
                
                job_status = status_data['generations_by_pk']['status']
                
                if job_status == 'COMPLETE':
                    images = status_data['generations_by_pk']['generated_images']
                    if images:
                        return {
                            "url": images[0]['url'],
                            "source": "Leonardo.ai (Premium)",
                            "prompt": prompt,
                            "provider": "leonardo"
                        }
            
            return {"error": "Leonardo Timed Out"}
            
        except Exception as e:
            return {"error": f"Leonardo Exception: {str(e)}"}

    def _construct_prompt(self, location, risk, chem, context=None):
        """
        Creates a prompt.
        """
        visual_state = "clear blue water"
        if "High" in risk or "CRITICAL" in risk:
            visual_state = "murky green water, bioluminescent algae streaks, thick surface scum"
        elif "Moderate" in risk:
            visual_state = "slightly turbid water, patches of green foam"
            
        chem_desc = f"Nitrate levels {chem.get('nitrate', 0)}, Dissolved Oxygen {chem.get('dissolved_oxygen', 0)}"
        
        base_prompt = (
            f"Satellite photography of {location}, ocean surface, {visual_state}. "
            f"Aerial view, 8k resolution, photorealistic, cinematic lighting. "
            f"Scientific visualization of algae bloom, high detail, water texture. "
        )
        
        if context:
            base_prompt += f" Tactical Focus: {context}. Highlight specific features related to this query."
            
        return base_prompt

    def generate_custom_image(self, custom_prompt, model="flux"):
        """
        Directly generates an image from a custom user prompt, routing to the correct provider.
        """
        # LEONARDO ROUTING
        if self.leo_key:
            # Re-use Leonardo logic but override prompt
            # We can't reuse generate_with_leonardo easily because it constructs its own prompt.
            # So we duplicate the request logic for custom prompts here for simplicity.
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.leo_key}"
            }
            
            payload = {
                "height": 576,
                "width": 1024,
                "prompt": custom_prompt,
                "modelId": "e316348f-7773-490e-adcd-46757c738eb7", # Absolute Reality v1.6
                "num_images": 1
            }
            
            try:
                response = requests.post(self.leo_url, json=payload, headers=headers)
                if response.status_code != 200:
                    return {"error": f"Leonardo Custom Init Failed: {response.text}"}
                    
                generation_id = response.json()['sdGenerationJob']['generationId']
                
                # Poll
                for _ in range(10):
                    time.sleep(2)
                    check_url = f"{self.leo_url}/{generation_id}"
                    status_resp = requests.get(check_url, headers=headers)
                    status_data = status_resp.json()
                    if status_data['generations_by_pk']['status'] == 'COMPLETE':
                        images = status_data['generations_by_pk']['generated_images']
                        if images:
                            return {
                                "url": images[0]['url'],
                                "source": "Leonardo.ai (Premium Custom)",
                                "prompt": custom_prompt,
                                "provider": "leonardo"
                            }
                return {"error": "Leonardo Custom Timed Out"}
            except Exception as e:
                return {"error": f"Leonardo Error: {str(e)}"}

        # POLLINATIONS ROUTING (Default)
        target_model = model
        final_prompt = custom_prompt
        
        # Enhance "flux-realism"
        if model == "flux-realism":
            target_model = "flux" # Use base flux
            final_prompt = f"{custom_prompt}, raw photo, hyper-realistic, 8k, highly detailed, fujifilm, cinematic lighting"

        # Enhance "flux-3d"
        if model == "flux-3d":
            target_model = "flux"
            final_prompt = f"{custom_prompt}, 3D render, unreal engine 5, volumetric lighting, octane render, isometric, c4d"
        
        # Sanitize
        safe_prompt = final_prompt.replace("?", "").replace("&", "").replace("#", "")
        encoded = urllib.parse.quote(safe_prompt)
        
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=576&model={target_model}&nologo=true&seed={int(time.time())}"
        return {
            "url": url,
            "source": f"Pollinations.ai ({model})",
            "prompt": final_prompt,
            "provider": "pollinations"
        }
