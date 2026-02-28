from tools import ToolRegistry, MultimodalSearchTool


def main():
    print('Creating ToolRegistry...')
    registry = ToolRegistry()
    print('Registering MultimodalSearchTool...')
    registry.register(MultimodalSearchTool())

    tool = registry.get_tool("multimodal_search")
    if not tool:
        print("multimodal_search tool not found in registry")
        return

    print('Calling multimodal_search.execute(...)')
    result = tool.execute(query="RAG architecture diagram")

    print("=== Multimodal Search Test ===")
    print("success:", result.success)
    print("error:", result.error)
    print("data:")
    print(result.data)


if __name__ == "__main__":
    main()
