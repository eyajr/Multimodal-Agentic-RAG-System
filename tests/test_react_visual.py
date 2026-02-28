from rag import initialize_system


def main():
    orchestrator, memory = initialize_system()

    query = "Show me the RAG architecture diagram"
    print(f"Routing query: {query}\n")

    # Force using ReAct agent
    result = orchestrator.route_task(query, preferred_agent='react')

    print("=== ReAct Visual Test Result ===")
    print(result)


if __name__ == '__main__':
    main()
