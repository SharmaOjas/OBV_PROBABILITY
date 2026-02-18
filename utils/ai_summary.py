from groq import Groq


def get_ai_summary(api_key, results_data):
    """
    Generates a summary of the scan results using Groq API.
    """
    client = Groq(api_key=api_key)

    # Prepare data for the prompt
    # We'll take the top 10 results to avoid token limits if list is huge
    top_results = results_data[:15]

    prompt_content = f"""
    You are a financial analyst assistant. 
    Here are the findings from an OBV Divergence Scan on Indian Stocks:

    {top_results}

    Please provide a concise but insightful summary of these findings.
    Highlight the most significant Bullish and Bearish setups.
    Explain what these divergences might indicate for the short-term trend of these specific stocks.
    Format the output in clean Markdown.
    """

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    return completion.choices[0].message.content

