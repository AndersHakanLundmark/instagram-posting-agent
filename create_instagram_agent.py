import os
import sys
import json
import argparse
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Try to load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loader fallback
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v.strip('"\'')

INSTAGRAM_SYSTEM_PROMPT = """
Persona: You are Navichain's Instagram Strategist. Your style is visual, inspiring, and "clean". You know that Instagram is about stopping the scroll with strong images and insightful, digestible captions.

Task: Create an Instagram caption based on the article's URL.

LAYOUT & FORMATTING (IMPORTANT):
1.  **No Links:** You know that links DO NOT work in captions. NEVER write out the URL in the text.
2.  **CTA:** ALWAYS use "Link in bio" (or "Länk i bio" in Swedish).
3.  **Air/Spacing:** Use line breaks to make the text airy (optionally use a dot "." on empty lines if required for formatting, but preferably just empty lines).
4.  **Emojis:** Use to reinforce the feeling.
5.  **Language:** Match the article's language (sv/en).

CONTENT:
1.  **The Hook:** A short sentence relating to the image (even if you don't see it, write as if the image illustrates the topic).
2.  **Storytelling:** Tell a micro-story or provide a strong insight from the article.
3.  **Value Drop:** "Here are three things you need to know:" followed by a bulleted list.
4.  **Hashtags:** A solid block of 10-20 relevant hashtags at the bottom, separated from the text by line breaks.

IMPORTANT:
- Focus on "aesthetic text". It should look good.
- Drive traffic to the profile (Bio).
"""

HISTORY_FILE = "instagram_post_history.json"

def load_history():
    """Laddar historik över använda vinklar/hooks för URLer."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_history(history):
    """Sparar uppdaterad historik."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save history: {e}", file=sys.stderr)


def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        # Fallback for common variable names
        key = os.getenv("GOOGLE_API_KEY")
    return key

def scrape_article(url):
    try:
        headers = {'User-Agent': 'NavichainInstagramAgent/1.0 (Business Intelligence Scraper)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Title
        title = soup.title.string if soup.title else ""
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        elif not title:
             title = "Untitled Article"

        # Headers Structure
        structure = []
        for header in soup.find_all(['h2', 'h3']):
            structure.append(header.get_text(strip=True))
            
        # Body text (paragraphs)
        paragraphs = []
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 40: # Filter short snippets/navigation/footers
                paragraphs.append(text)
        
        full_text = "\n\n".join(paragraphs)
        
        # Featured Image (Open Graph)
        image_url = ""
        og_image = soup.find('meta', property='og:image')
        if og_image:
            image_url = og_image.get('content')
        else:
            # Fallback twitter image
            tw_image = soup.find('meta', attrs={'name': 'twitter:image'})
            if tw_image:
                image_url = tw_image.get('content')
        
        # OG Title (often better than <title>)
        og_title = soup.find('meta', property='og:title')
        if og_title:
             title = og_title.get('content')
             
        # Force HTTPS on image URL
        if image_url and image_url.startswith('http://'):
            image_url = image_url.replace('http://', 'https://')

        # Scrape Tags/Keywords (Frontmatter equivalent)
        tags = []
        # 1. Keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            tags.extend([t.strip() for t in meta_keywords.get('content').split(',')])
        
        # 2. Article Tags (Open Graph / Facebook)
        for tag in soup.find_all('meta', property='article:tag'):
            if tag.get('content'):
                tags.append(tag.get('content').strip())
                
        # Deduplicate
        tags = list(set(filter(None, tags)))

        return {
            "url": url,
            "title": title,
            "structure": structure,
            "content": full_text[:15000], # Cap strictly at 15k chars for token limits
            "image_url": image_url,
            "source_tags": tags
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}", file=sys.stderr)
        return None

def generate_instagram_post(article_data, api_key, history=None):
    genai.configure(api_key=api_key)
    # Using flash-lite for speed and efficiency
    model_name = 'gemini-2.0-flash-lite' 
    model = genai.GenerativeModel(model_name)
    
    target_lang = "sv"
    if "en-gb" in article_data['url'] or "/en/" in article_data['url'] or "-en-" in article_data['url']:
        target_lang = "en"
    
    # Check history for previous angles
    previous_angles_prompt = ""
    if history and article_data['url'] in history:
        previous_angles = history[article_data['url']]
        if previous_angles:
            previous_angles_prompt = f"""
    PREVIOUSLY USED ANGLES (DO NOT REPEAT THESE):
    The following angles/hooks have already been used for this article. You MUST choose a completely different perspective, theme, or angle.
    {json.dumps(previous_angles, indent=2)}
    
    INSTRUCTION: Generate a FRESH perspective that is distinct from the above.
    """

    
    user_prompt = f"""
    ANALYSIS TARGET:
    URL: {article_data['url']}
    TITLE: {article_data['title']}
    IMAGE: {article_data['image_url']}
    SOURCE TAGS (OPTIMIZE THESE): {', '.join(article_data.get('source_tags', []))}
    TARGET LANGUAGE: {target_lang} (The tweet MUST be in this language)
    
    STRUCTURE (Headlines):
    {json.dumps(article_data['structure'], indent=2)}
    
    {previous_angles_prompt}
    
    CONTENT EXCERPT:
    {article_data['content']}
    
    TASK: Generate the Instagram JSON object now.
    
    IMPORTANT: The JSON response MUST follow this schema:
    {{
      "instagram_post": {{
        "caption_text": "The full caption text including hook, insight/list, CTA (Link in bio), and hashtags block.",
        "angle_description": "A very brief (1 sentence) description of the specific angle/theme used in this post, so we can avoid it next time."
      }}
    }}
    """
    
    try:
        response = model.generate_content(
            f"{INSTAGRAM_SYSTEM_PROMPT}\n\n{user_prompt}",
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        
        # Normalize
        if 'post_text' not in data:
            content_source = data.get('instagram_post', data)
            if isinstance(content_source, dict) and 'caption_text' in content_source:
                 data['post_text'] = content_source['caption_text']
            
        # Post-processing: Clean up text artifacts
        if 'post_text' in data:
            clean = data['post_text']
            
            # Clean up potential markdown or labels
            garbage = ["Caption Text:", "CAPTION:", "**"]
            for g in garbage:
                clean = clean.replace(g, "")
            
            # Ensure no raw URLs in text (Instagram doesn't link them)
            # If the specific URL is found, replace/remove it to keep it clean, unless it's a "Link in bio" context.
            # But the prompt said NO links. The AI might slip up.
            if article_data['url'] in clean:
                 clean = clean.replace(article_data['url'], "")
            
            clean = clean.strip()
            
            data['post_text'] = clean
            data['language'] = target_lang 
        
        return data

    except Exception as e:
        print(f"Error generating AI content: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Navichain Instagram Agent")
    parser.add_argument("url", help="URL of the article to process")
    parser.add_argument("--webhook", default="https://n8n.navichain.se/webhook/instagram-post", help="n8n Webhook URL to send JSON to")
    parser.add_argument("--dry-run", action="store_true", help="Only print JSON, do not send to webhook even if provided")
    args = parser.parse_args()
    
    # 1. Setup
    api_key = get_api_key()
    if not api_key:
        print("CRITICAL ERROR: GEMINI_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)
        
    # 2. Scrape
    print(f"Processing Article: {args.url}...", file=sys.stderr)
    data = scrape_article(args.url)
    if not data:
        print("Failed to scrape article.", file=sys.stderr)
        sys.exit(1)
        
    print(f"  > Found Title: {data['title']}", file=sys.stderr)
    
    # Load history
    history = load_history()
    
    # 3. Generate
    print("Generating Instagram Content...", file=sys.stderr)
    result = generate_instagram_post(data, api_key, history)
    
    if not result:
        print("Failed to generate content.", file=sys.stderr)
        sys.exit(1)

    # 4. Refine/Validate
    if not result.get('article_url'):
        result['article_url'] = args.url
    
    if not result.get('image_url') or result['image_url'] == "":
        result['image_url'] = data['image_url']
        
    if not result.get('title'):
        result['title'] = data['title']

    # 5. Output
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    print(json_output)
    
    # Save varied angle to history
    if result and args.url:
        new_angle = ""
        post_data = result.get('instagram_post') or result
        if isinstance(post_data, dict):
             new_angle = post_data.get('angle_description')
        
        if new_angle:
            if args.url not in history:
                history[args.url] = []
            
            if new_angle not in history[args.url]:
                history[args.url].append(new_angle)
                save_history(history)
                print(f"  > Saved new angle to history: '{new_angle}'", file=sys.stderr)
    
    # 6. Webhook
    if args.webhook and not args.dry_run:
        print(f"Transmitting to n8n Webhook: {args.webhook}...", file=sys.stderr)
        try:
            resp = requests.post(args.webhook, json=result)
            if resp.status_code in [200, 201, 204]:
                 print("  > Success: Webhook received payload.", file=sys.stderr)
            else:
                 print(f"  > Error: Webhook returned status {resp.status_code}: {resp.text}", file=sys.stderr)
        except Exception as e:
            print(f"  > Connection Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
