import time
from datetime import datetime, timedelta, timezone
from collections import deque
from gdeltdoc import GdeltDoc, Filters
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize GDELT client
gd = GdeltDoc()

# Track recently seen article URLs
seen_articles = deque(maxlen=100)  # Rolling cache with a max size of 100 normalized URLs

# Track the most recent article's publication time
last_seen_time = datetime.now(timezone.utc) - timedelta(minutes=2)

# Load Mistral 7B model and tokenizer
print("Loading Mistral 7B model...")
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map to GPU if available
    low_cpu_mem_usage=True,  # Optimize memory usage
)

# Counter for headlines to trigger test injection
headline_counter = 0



def analyze_headline(headline):
    """
    Analyze the headline using Mistral 7B to determine if it mentions a Russian attack on NATO.
    """
    # Refined prompt with more explicit examples
    prompt = (
        "Classify the headline below as 'Yes' if it says Russia has attacked or conducted military action in a NATO country, "
        "or 'No' if it does not. Only respond with 'Yes' or 'No'.\n\n"
        "Examples:\n"
        "Headline: Russia launches a missile strike on NATO base in Poland\nAnswer: Yes\n"
        "Headline: Russian forces cross into Latvia (a NATO country)\nAnswer: Yes\n"
        "Headline: Russian navy conducts drills in the Arctic Ocean\nAnswer: No\n\n"
        f"Headline: {headline}\nAnswer:"
    )

    # Tokenize and preprocess the prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Generate a response from the model
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,  # Limit output to 5 tokens
        do_sample=False,  # Deterministic decoding
        pad_token_id=tokenizer.pad_token_id,  # Explicitly set pad token
        eos_token_id=tokenizer.eos_token_id,  # Ensure generation stops at EOS
    )

    # Decode the generated output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Generated Output: {result}")  # Debugging

    # Extract and validate the answer
    if "Answer:" in result:
        answer = result.split("Answer:")[-1].strip().split()[0]  # Get the first word after "Answer:"
        if answer in ["Yes", "No"]:
            return answer

    return "Unknown"  # Default if no valid answer is found




# Inject a test headline to verify positive case detection
test_headline = "Russia wins chess battle after 120 moves"
print(f"\nInjected Test Headline: {test_headline}")
test_analysis = analyze_headline(test_headline)
print(f"Test Analysis: {test_analysis}")
print("=" * 40)
time.sleep(10)


while True:
    print(f"\nPolling GDELT API at {datetime.now(timezone.utc).isoformat()}...")

    # Define filters with a longer, generic keyword
    filters = Filters(
        keyword="breaking news",  # Broad query to fetch more articles
        start_date=last_seen_time.strftime("%Y-%m-%d"),
        end_date=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    )

    try:
        # Fetch articles
        articles_df = gd.article_search(filters)

        # Check if DataFrame is empty
        if articles_df.empty:
            print("No articles found.")
        else:
            print(f"Found {len(articles_df)} articles.")
            # Display each new article and analyze with Mistral 7B
            for index, row in articles_df.iterrows():
                raw_url = row.get("url")
                raw_title = row.get("title", "No Title")
                # Normalize the URL
                if raw_url:
                    normalized_url = raw_url.strip().lower().rstrip('/')
                else:
                    normalized_url = None

                # Increment the headline counter
                headline_counter += 1

                # Inject a test headline every 10 headlines
                if headline_counter % 10 == 0:
                    test_headline = "Russia launches a missile strike on NATO base in Poland"
                    print(f"\nInjected Test Headline: {test_headline}")
                    test_analysis = analyze_headline(test_headline)
                    print(f"Test Analysis: {test_analysis}")
                    print("=" * 40)

                    # Verify and skip the test headline
                    if test_analysis != "Yes":
                        print("Warning: Test headline did not return expected 'Yes'")
                    continue

                if normalized_url and normalized_url not in seen_articles:
                    # Display the article
                    print(f"Title: {raw_title}")
                    print(f"URL: {raw_url}")
                    print(f"Source: {row.get('sourcecountry', 'Unknown')}")
                    print("-" * 40)

                    # Analyze the headline with Mistral 7B
                    analysis = analyze_headline(raw_title)
                    print(f"Mistral Analysis: {analysis}")
                    print("=" * 40)

                    # Add the normalized URL to the queue
                    seen_articles.append(normalized_url)

    except Exception as e:
        print(f"Error during polling: {e}")

    # Wait for 1 minute before polling again
    time.sleep(60)
