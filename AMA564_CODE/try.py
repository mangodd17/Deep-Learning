import pandas as pd
import csv
from openai import OpenAI
import time
import os

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="key",  
)

def generate_answer(question, max_retries=3):
    """
    Use GPT-4.1 model to generate question answers with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            # Extract the actual question content, removing the role-playing instruction if it exists
            if "请你扮演一位金融和会计领域专家" in question:
                # Find where the actual question starts after the role instruction
                parts = question.split("请你扮演一位金融和会计领域专家，你会面临用户提出的一些问题，你要给出解决问题的思考过程和最终答案。你要首先在头脑中思考推理过程，然后向用户提供答案。最后，答案要用 $\\boxed{答案}$的形式输出。")
                if len(parts) > 1:
                    actual_question = parts[1].strip()
                else:
                    actual_question = question
            else:
                actual_question = question
            
            # Clean up the question text
            actual_question = actual_question.replace('\\n', '\n').strip()
            
            # Optimized prompt for GPT-4.1
            prompt = f"""Please solve this financial problem step by step:
{actual_question}

Requirements:
- If the question is in Chinese, output in Chinese; if the question is in English, answer in English
- Provide clear reasoning process
- Show all calculation steps
- If this is a multiple choice question, output all correct option letters (A, B, C, D), NOT numbers
- Give final answer in $\\boxed{{answer}}$ format

Please be thorough but concise in your explanation."""

            # Call GPT-4.1 API
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://your-site.com",  # Optional
                    "X-Title": "Financial Problem Solver"     # Optional
                },
                extra_body={},
                model="openai/gpt-4.1",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=8000  
            )
            
            # Check if response is valid
            if completion and completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                if content and len(content.strip()) > 0:
                    # Clean up boxed format - remove \text{} wrapper
                    cleaned_content = content.replace('\\boxed{\\text{', '\\boxed{').replace('}}', '}')
                    # Also handle cases with quotes
                    cleaned_content = cleaned_content.replace('\\boxed{"', '\\boxed{').replace('"}', '}')
                    return cleaned_content
                else:
                    raise Exception("Empty response from API")
            else:
                raise Exception("Invalid response structure from API")
        
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed: {error_msg}")
            
            # For persistent errors, try simplified request
            if "Expecting value" in error_msg or "timeout" in error_msg.lower():
                try:
                    # Simplified request for GPT-4.1
                    simple_completion = client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://your-site.com",
                            "X-Title": "Financial Problem Solver"
                        },
                        extra_body={},
                        model="openai/gpt-4.1",
                        messages=[
                            {
                                "role": "user", 
                                "content": f"Solve this financial problem.  If this is a multiple choice question, output all correct option letters (A, B, C, D)  in $\\boxed{{}}$ format: {actual_question[:600]}"
                            }
                        ],
                        max_tokens=6000,
                        temperature=0.1
                    )
                    
                    if simple_completion and simple_completion.choices:
                        simple_content = simple_completion.choices[0].message.content
                        if simple_content:
                            # Clean up format
                            cleaned_simple = simple_content.replace('\\boxed{\\text{', '\\boxed{').replace('}}', '}')
                            cleaned_simple = cleaned_simple.replace('\\boxed{"', '\\boxed{').replace('"}', '}')
                            return cleaned_simple
                except Exception as simple_e:
                    print(f"Simple attempt also failed: {simple_e}")
            
            if attempt < max_retries - 1:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)  # Adjusted delay for GPT-4.1
            else:
                # Return a structured error that can still be processed
                return f"Due to persistent API errors, unable to process this question. Error: {error_msg[:100]}"
    
    return "Error: Maximum retries exceeded due to API issues"

def process_input_csv(input_file_path, output_file_path, max_questions=None):
    """
    Process input CSV file and generate answers using GPT-4.1
    max_questions: limit the number of questions to process, None means process all
    """
    # Read input file with proper encoding and error handling
    try:
        # Try different encodings and separators
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(input_file_path, header=None, encoding=encoding, sep=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
                break
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                try:
                    df = pd.read_csv(input_file_path, header=None, encoding=encoding, sep=None, engine='python')
                    break
                except:
                    continue
        
        if df is None:
            raise Exception("Could not read CSV file with any encoding")
        
        if max_questions is not None:
            df = df.head(max_questions)
            
    except Exception as e:
        print(f"Failed to read input file: {e}")
        return
    
    # Prepare output data
    output_data = []
    
    # Process each row
    for index, row in df.iterrows():
        try:
            # Handle different CSV formats
            if len(row) >= 3:
                # Format: column_number, question_id, question_content
                column_number = row[0]
                question_id = row[1]
                question = row[2]
            elif len(row) == 2:
                # Format: question_id, question_content
                column_number = index
                question_id = row[0]
                question = row[1]
            else:
                print(f"Row {index} has unexpected format, skipping")
                continue
                
            print(f"Processing question {question_id} with GPT-4.1...")
            
            # Generate answer
            answer = generate_answer(str(question))
            
            # Add to output data
            # Format: column_number, question_id, sample_id, model_output
            output_data.append([column_number, question_id, 0, answer])
            
            print(f"Question {question_id} completed")
            
            # Add delay to avoid API limits (adjusted for GPT-4.1)
            time.sleep(1)
                
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            # Add error record
            error_question_id = row[1] if len(row) > 1 else f"row_{index}"
            output_data.append([index, error_question_id, 0, f"Processing error: {str(e)}"])
    
    # Save output file
    try:
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in output_data:
                writer.writerow(row)
        
        print(f"Output saved to {output_file_path}")
        
    except Exception as e:
        print(f"Failed to save output file: {e}")

def main():
    """
    Main function
    """
    input_file = "input.csv"
    output_file = "output_gpt41.csv"
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist!")
        return
    
    print("Starting processing with GPT-4.1...")
    process_input_csv(input_file, output_file)
    print("Processing completed!")

if __name__ == "__main__":
    main()
