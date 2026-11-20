from gradio_client import Client

client = Client("agents-course/decoding_visualizer")
result = client.predict(
    input_text="The Capital of France is",
    api_name="/get_beam_search_html"
)

print("Result 1:\n", result[0])
print("\nResult 2:\n", result[1])
