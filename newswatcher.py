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

# Load GPT-Neo-2.7B model and tokenizer
print("Loading GPT-Neo-2.7B model...")
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map to available hardware
    low_cpu_mem_usage=True,  # Optimize memory usage
)

def analyze_headline(headline):
    """
    Analyze the headline to determine if it mentions nuclear weapons or NATO strikes.
    """
    prompt = (
        f"Does the following headline mention Russia using nuclear weapons or striking a NATO target? "
        f"Answer only 'Yes' or 'No'.\n\n"
        f"Headline: {headline}"
    )
    # Tokenize and preprocess the prompt
    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=512,  # Truncate to 512 tokens
        padding="max_length",  # Pad to maximum length
        return_tensors="pt"  # Return PyTorch tensors
    )

    # Move inputs to the correct device
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=3,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Control randomness
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Extract "Yes" or "No" from the result
    if "Yes" in result:
        return "Yes"
    elif "No" in result:
        return "No"
    else:
        return "Unknown"  # Fallback if the response isn't as expected


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
            # Display each new article and analyze with GPT-Neo-2.7B
            for index, row in articles_df.iterrows():
                raw_url = row.get("url")
                raw_title = row.get("title", "No Title")
                # Normalize the URL
                if raw_url:
                    normalized_url = raw_url.strip().lower().rstrip('/')
                else:
                    normalized_url = None

                if normalized_url and normalized_url not in seen_articles:
                    # Display the article
                    print(f"Title: {raw_title}")
                    print(f"URL: {raw_url}")
                    print(f"Source: {row.get('sourcecountry', 'Unknown')}")
                    print("-" * 40)

                    # Analyze the headline with GPT-Neo-2.7B
                    analysis = analyze_headline(raw_title)
                    print(f"GPT-Neo Analysis: {analysis}")
                    print("=" * 40)

                    # Add the normalized URL to the queue
                    seen_articles.append(normalized_url)

    except Exception as e:
        print(f"Error during polling: {e}")

    # Wait for 1 minute before polling again
    time.sleep(60)
