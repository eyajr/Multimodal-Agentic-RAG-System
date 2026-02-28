from rag import initialize_system


def main():
    orchestrator, memory = initialize_system()
    print('Memory before clear:')
    print('Messages:', len(memory.messages))
    print('Thoughts:', len(memory.agent_thoughts))

    memory.clear()

    print('\nMemory after clear:')
    print('Messages:', len(memory.messages))
    print('Thoughts:', len(memory.agent_thoughts))


if __name__ == '__main__':
    main()
