import time
import re
from datetime import datetime, timedelta, timezone
from collections import deque
from gdeltdoc import GdeltDoc, Filters
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize GDELT client
gd = GdeltDoc()

# Track recently seen article URLs
seen_articles = deque(maxlen=500)  # Increased cache size for deduplication

# Track the most recent article's publication time
last_seen_time = datetime.now(timezone.utc) - timedelta(minutes=2)

# Load Mistral 7B model and tokenizer
print("Loading Mistral 7B model...")
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="float16",  # Use half-precision for faster inference
    low_cpu_mem_usage=True,
)






def analyze_headlines_block(headlines, model, tokenizer):
    """
    Analyze a block of headlines using Mistral 7B.
    """
    # Limit to the maximum number of headlines that fit within the model's context window
    max_headlines = 100
    headlines = headlines[:max_headlines]

    # Construct the prompt
    prompt = (
        "Analyze the following news headlines. If any headline suggests Russia attacking a NATO country "
        "or starting military conflicts involving NATO, respond with 'Yes' followed by the troubling headline(s). "
        "If none of them suggest such activity, respond with 'No'. Respond only with 'Yes' or 'No'.\n\n"
        "Headlines:\n" + "\n".join(f"- {headline}" for headline in headlines) + "\n\nResponse:"
    )

    # Tokenize the block of text
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to("cuda")

    # Generate a response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,  # Strictly limit output length
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the response
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Debugging: Display the raw output
    print(f"Raw Model Output:\n{result}\n")

    # Validate and parse the response
    if result.startswith("Yes"):
        troubling_headlines = re.findall(r"- .*", result)
        troubling_text = "\n".join(troubling_headlines)
        return f"Yes. {troubling_text}" if troubling_text else "Yes."
    elif result.startswith("No"):
        return "No"
    else:
        # Fallback for unexpected responses
        return f"Unknown. Model response:\n{result}"





# Inject a short test block at the start
test_headlines = [
    "Russian forces cross into Latvia",
    "Russia launches a missile strike on NATO base in Poland",
    "Local sports team wins championship",
]
print(f"\nInjected Test Block:\n{test_headlines}")
test_response = analyze_headlines_block(test_headlines, model, tokenizer)
print(f"Test Block Analysis:\n{test_response}")
print("=" * 40)

# Main loop remains the same
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

            # Gather new headlines
            headlines_to_analyze = []
            for index, row in articles_df.iterrows():
                raw_url = row.get("url")
                raw_title = row.get("title", "No Title")
                # Normalize the URL
                if raw_url:
                    normalized_url = raw_url.strip().lower().rstrip('/')
                else:
                    normalized_url = None

                # Skip already seen articles
                if normalized_url and normalized_url not in seen_articles:
                    headlines_to_analyze.append(raw_title)
                    seen_articles.append(normalized_url)

            # Analyze the block of headlines
            if headlines_to_analyze:
                print("\nAnalyzing Block of Headlines...")
                response = analyze_headlines_block(headlines_to_analyze, model, tokenizer)
                print(f"Model Response:\n{response}")
                print("=" * 40)

    except Exception as e:
        print(f"Error during polling: {e}")

    # Wait for 1 minute before polling again
    time.sleep(60)
