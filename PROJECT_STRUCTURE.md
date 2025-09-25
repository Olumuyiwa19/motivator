# Project Structure

This document outlines the directory structure for the Faith-based Motivator Chatbot AWS deployment.

## Root Directory

```
chatbot-aws-deployment/
├── .env                          # Environment variables (not in git)
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Main Python dependencies
├── docker-compose.yml            # Local development with Docker
├── setup.py                      # Project setup script
├── PROJECT_STRUCTURE.md          # This file
└── README.md                     # Project documentation
```

## Infrastructure (AWS CDK)

```
infrastructure/
├── app.py                        # CDK app entry point
├── cdk.json                      # CDK configuration
├── requirements.txt              # CDK dependencies
└── stacks/
    ├── __init__.py
    └── chatbot_stack.py          # Main infrastructure stack
```

## Backend (FastAPI)

```
backend/
├── main.py                       # FastAPI application entry point
├── models.py                     # Pydantic models for API
├── requirements.txt              # Backend dependencies
├── Dockerfile                    # Container configuration
└── services/
    ├── __init__.py
    ├── chatbot_service.py        # Core chatbot logic
    └── auth_service.py           # Authentication service
```

## Frontend (Reflex)

```
frontend/
├── chatbot_app.py                # Main Reflex application
├── state.py                      # Reflex state management
├── requirements.txt              # Frontend dependencies
├── rxconfig.py                   # Reflex configuration
├── Dockerfile                    # Container configuration
└── components/
    ├── __init__.py
    ├── chat_interface.py         # Chat UI components
    └── auth_components.py        # Authentication UI components
```

## Key Features

### Infrastructure as Code (CDK)

- **Python-based CDK** for infrastructure definition
- **Environment-specific stacks** for dev/staging/production
- **Modular stack design** for maintainability

### Backend API (FastAPI)

- **RESTful API** with OpenAPI documentation
- **Pydantic models** for request/response validation
- **Service layer architecture** for business logic separation
- **JWT authentication** integration with Cognito
- **Docker containerization** for consistent deployment

### Frontend Application (Reflex)

- **Pure Python frontend** with Reflex framework
- **Component-based architecture** similar to React
- **State management** with Reflex State
- **Responsive design** with modern UI components
- **Authentication integration** with Cognito

### Development Environment

- **Virtual environment** support for Python dependencies
- **Docker Compose** for local development
- **Environment configuration** with .env files
- **Git integration** with comprehensive .gitignore

## Development Workflow

1. **Setup**: Run `python setup.py` to initialize the project
2. **Environment**: Configure `.env` file with AWS credentials
3. **Backend Development**: Work in `backend/` directory
4. **Frontend Development**: Work in `frontend/` directory
5. **Infrastructure**: Define resources in `infrastructure/stacks/`
6. **Testing**: Use Docker Compose for integration testing
7. **Deployment**: Use CDK for AWS deployment

## Next Implementation Tasks

The project structure is now ready for implementing the remaining tasks:

- **Task 2**: Extract Streamlit logic to FastAPI backend
- **Task 3**: Complete Docker containerization
- **Task 4**: Implement CDK infrastructure components
- **Task 5**: Deploy ECS Fargate service
- **Task 6**: Set up Cognito authentication
- **Task 7**: Complete Reflex frontend implementation
- **Tasks 8-14**: Security, monitoring, CI/CD, and optimization

Each component has placeholder implementations that will be completed in subsequent tasks.
