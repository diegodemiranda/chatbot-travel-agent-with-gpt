```markdown
# Travel Agent Application

## Overview
The Travel Agent application is designed to provide users with comprehensive travel itineraries and tips. Leveraging the power of OpenAI's GPT models, the application offers personalized travel advice, including event contexts, ticket prices, and relevant document retrieval to ensure a seamless travel planning experience.

## Features
- **Chat-Based Interface**: Utilizes OpenAI's ChatGPT for interactive travel queries.
- **Research Agent**: Employs various tools for fetching web-based travel information.
- **Document Retrieval**: Gathers and processes relevant travel documents to enhance recommendations.
- **Supervisor Agent**: Integrates multiple data sources to generate comprehensive travel advice.

## Architecture
The application is structured around a serverless architecture, deployed on AWS Lambda for scalable and efficient request handling. It incorporates several components, including:
- **LangChain OpenAI Integration**: For accessing OpenAI's models.
- **Web-Based Data Loaders**: For scraping travel-related information from specified URLs.
- **Vector Storage and Retrieval**: Utilizes Chroma for document storage and retrieval based on query relevance.

## Deployment
The application is containerized using Docker, facilitating easy deployment to AWS. The provided `Dockerfile` outlines the setup process, including environment preparation and dependency installation.

### Prerequisites
- AWS Account
- Docker installed on your local machine
- AWS CLI configured

### Steps
1. **Build the Docker Image**:
   ```sh
   docker build -t travel-agent-app .
   ```
2. **Push the Image to Amazon ECR**:
   Follow AWS documentation to create a repository and push the Docker image.
3. **Deploy to AWS Lambda**:
   Use the AWS Management Console or AWS CLI to create a Lambda function with the uploaded Docker image.

## Configuration
The application requires an OpenAI API key for operation. This key should be set in a `.env` file in the project root or configured as an environment variable in AWS Lambda.

## Usage
Invoke the Lambda function with a JSON payload containing a `question` key for travel-related queries. The application will process the request and return a comprehensive travel guide as a response.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or documentation improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- Diego de Miranda

## Acknowledgments
- OpenAI for providing the GPT models.
- The LangChain community for the agent toolkits and integration libraries.
```

This README provides a basic overview of the Travel Agent application, its features, architecture, deployment instructions, usage, and contribution guidelines. Adjustments may be needed based on the specific AWS setup and any additional features or changes made to the application.
