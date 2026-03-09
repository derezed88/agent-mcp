import requests
import json
import time

base_url = "http://localhost:8765"

# First, discover what endpoints are available
def discover_endpoints():
    """Make a few test calls to discover API"""
    endpoints_to_test = [
        "/api/v1/delegate",
        "/api/v1/execute",
        "/api/v1/query",
        "/v1/delegate",
        "/delegate"
    ]
    
    for endpoint in endpoints_to_test:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.post(url, json={"prompt": "test", "mode": "tool"}, timeout=5)
            print(f"\n{endpoint}: Status {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.text[:100]}")
        except Exception as e:
            print(f"{endpoint}: Error - {e}")

def call_api(prompt, mode="reasoning", delegatee="samaritan-execution"):
    """Make API call to localhost:8765"""
    # Try multiple endpoints
    for endpoint in ["/api/v1/delegate", "/api/v1/execute", "/delegate"]:
        url = f"{base_url}{endpoint}"
        data = {"delegatee": delegatee, "prompt": prompt, "mode": mode}
        try:
            response = requests.post(url, json=data, timeout=30)
            print(f"\n=== Endpoint: {endpoint} ===")
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": response.status_code, "message": response.text[:200]}
        except Exception as e:
            print(f"Error: {e}")
    return {"error": "All endpoints failed"}

def main():
    """Loop and call the API"""
    print("API Client started on localhost:8765")
    
    # First discover
    print("\n=== Discovering endpoints ===")
    discover_endpoints()
    
    # Then test loop
    print("\n=== Starting test loop ===")
    for i in range(10):
        prompt = f"Loop iteration {i}: Get system time, query qwen_base"
        result = call_api(prompt)
        print(json.dumps(result, indent=2))
        time.sleep(2)
    
    print("\nTest loop complete")

if __name__ == "__main__":
    main()
