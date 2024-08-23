from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Initialize LLM
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

expert_template = """
You are Expert {expert_num}, one of three experts working on solving the following problem:
{problem}

Consider these factors: {factors}

Current step: {step}
Your previous thoughts: {previous_thoughts}
Other experts' thoughts from the previous step: {other_thoughts}

Provide the next step in your thinking process. If you think your approach is no longer viable based on the insights shared, respond with "I withdraw."

Your response:
"""

expert_prompt = ChatPromptTemplate.from_template(expert_template)
expert_chain = expert_prompt | llm | StrOutputParser()


def tree_of_thoughts(problem, factors, max_steps=3):
    experts = {1: [], 2: [], 3: []}
    active_experts = set([1, 2, 3])

    for step in range(1, max_steps + 1):
        print(f"\nStep {step}:")

        # Create inputs for each expert
        inputs = {
            f"expert_{expert}": {
                "expert_num": expert,
                "problem": problem,
                "factors": factors,
                "step": step,
                "previous_thoughts": "\n".join(experts[expert]),
                "other_thoughts": "\n".join([
                    f"Expert {other_expert}: {experts[other_expert][-1] if experts[other_expert] else 'No thoughts yet'}"
                    for other_expert in active_experts if other_expert != expert
                ])
            }
            for expert in active_experts
        }

        print(inputs)
        # Create and invoke RunnableParallel
        expert_chains = {
            f"expert_{expert}": lambda input: expert_chain.invoke(input[f"expert_{expert}"])
            for expert in active_experts
        }

        parallel_runnable = RunnableParallel(**expert_chains)
        results = parallel_runnable.invoke(inputs)
        # Process results from experts
        for expert_key, result in results.items():
            expert = int(expert_key.split('_')[1])
            thought = result
            print(f"Expert {expert}: {thought}")

            if thought.strip().lower() == "i withdraw.":
                active_experts.remove(expert)
                print(f"Expert {expert} has withdrawn.")
            else:
                experts[expert].append(thought)

        if len(active_experts) == 0:
            print("All experts have withdrawn. Ending the process.")
            break

        if len(active_experts) == 1:
            final_expert = list(active_experts)[0]
            print(f"\nOnly Expert {final_expert} remains. Final thought:")
            print(experts[final_expert][-1])
            break

    return experts


if __name__ == "__main__":
    problem = "How can we reduce carbon emissions in urban areas?"
    factors = "cost-effectiveness, technological feasibility, social impact, and implementation time"

    results = tree_of_thoughts(problem, factors)

    print("\nFinal Results:")
    for expert, thoughts in results.items():
        print(f"\nExpert {expert}'s thought process:")
        for i, thought in enumerate(thoughts, 1):
            print(f"Step {i}: {thought}")
