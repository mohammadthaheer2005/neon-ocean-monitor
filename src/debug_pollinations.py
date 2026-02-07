import requests
import time

url = "https://image.pollinations.ai/prompt/ocean_test?width=1024&height=576&model=flux&nologo=true"

print(f"Testing connectivity to: {url}")
try:
    start = time.time()
    response = requests.get(url, timeout=10)
    elapsed = time.time() - start
    
    print(f"Status Code: {response.status_code}")
    print(f"Time Taken: {elapsed:.2f}s")
    print(f"Content Type: {response.headers.get('Content-Type')}")
    print(f"Content Length: {len(response.content)} bytes")
    
    if response.status_code == 200 and len(response.content) > 1000:
        print("✅ SUCCESS: Received valid image data.")
        with open("debug_image.jpg", "wb") as f:
            f.write(response.content)
        print("Saved to debug_image.jpg")
    else:
        print("❌ FAILURE: Invalid response.")
        print("Response Text Preview:", response.text[:200])

except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
