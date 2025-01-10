# Define global variables
$global:PROJECT_NAME = "synthetic_pred"
$global:python_INTERPRETER = "python"

# Set up python interpreter environment
function create_environment {
    if (-Not (Test-Path -Path ".$global:PROJECT_NAME")) {
        & $global:python_INTERPRETER -m venv ".$global:PROJECT_NAME"
        Write-Output ">>> python sandbox environment: .$global:PROJECT_NAME created."
    }
    else {
        Write-Output ">>> python sandbox environment: .$global:PROJECT_NAME already exists"
    }
    Write-Output ">>> Activate python sandbox environment: $global:PROJECT_NAME using command: activate_environment"
}

# Activate python interpreter environment
function activate_environment {
    Write-Output ">>> Activating python sandbox environment: $global:PROJECT_NAME"
    try {
        & ".$global:PROJECT_NAME/Scripts/Activate.ps1"
        Write-Output ">>> Activated python sandbox environment: .$global:PROJECT_NAME"
        Write-Output ">>> Run 'deactivate' to exit the environment"
    }
    catch {
        Write-Output ">>> Error: Failed to activate python sandbox environment: .$global:PROJECT_NAME"
        Write-Output $_.Exception.Message
    }
}

# Install python Dependencies
function install_requirements {
    try {
        & $global:python_INTERPRETER -m pip install -r requirements.txt
        Write-Output ">>> python dependencies installed successfully"
    }
    catch {
        Write-Output ">>> Error: Failed to install python dependencies"
        Write-Output $_.Exception.Message
    }
}