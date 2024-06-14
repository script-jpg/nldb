from DataUtils.data_utils import DataUtils
from LLM.Impl.GPT3 import GPT3

if __name__ == "__main__":
    utils = DataUtils.get_default()
    utils.load('./Documents/utils.json', dict())
    llm = GPT3()
    while True:
        user_message = None

        try:
            user_message = input("Enter your message (or 'exit' to quit): ")
        except ValueError:
            continue
        if user_message.lower() == 'exit':
            break

        utils.update('./Documents')    
        relevant_data = utils.find_relevant(user_message)
        response = llm.get_response(f'Can you try to answer the query: "{user_message}", given the data: {relevant_data}')
        print("Response: ", response)
    utils.save('./Documents/utils.json')
        



