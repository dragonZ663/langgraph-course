from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    answer = app.invoke({"question": "What is Agent memory?"})
    print(answer)
