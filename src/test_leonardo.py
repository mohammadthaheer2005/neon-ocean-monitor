from image_gen_agent import OceanImageGenerator
import os

print("--- LEONARDO AI CONNECTION TEST ---")
api_key = input("Enter your Leonardo API Key: ").strip()

if not api_key:
    print("❌ No Key provided. Exiting.")
    exit()

print(f"✅ Key Received: {api_key[:5]}...*****")
print("Initializing Agent...")

agent = OceanImageGenerator(leonardo_key=api_key)

# Test Data
location = "Test Sector Alpha"
risk = "High Algae Bloom"
chem = {"nitrate": 5.0, "dissolved_oxygen": 3.2}

print("🚀 Sending generation request to Leonardo (this may take up to 20 seconds)...")
print("Prompting for: High Algae Bloom scenario...")

try:
    result = agent.generate_with_leonardo(location, risk, chem)
    
    if "error" in result:
        print(f"❌ TEST FAILED: {result['error']}")
    else:
        print("\n✅ TEST SUCCESSFUL!")
        print(f"📸 Image URL: {result['url']}")
        print(f"ℹ️ Provider: {result['source']}")
        print("Note: This URL is temporary. Open it immediately in your browser.")
        
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
