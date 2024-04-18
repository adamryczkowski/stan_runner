# NATS message breakup

## Task messages:

Each task message is a JSON object that is sent in the following subject:

stan.<model_hash>.<task_hash>.task

The JSON object has the following fields:

- task_hash: a unique identifier for the task - i.e. hash of everything needed to know except for this hash and model_name.
- model_code: the normalized model code. In the future we will use workers to normalize the model code - for now it is done by the scheduling agent (user).
- data: the data to be used in the model. This is a JSON object that is passed to the model code.
- model_parameters: the parameters to be used in the model. This is a JSON object that is passed to the model code.
- model_name: the name of the model. This is a string that is passed to the model code. Model name is not part of the task_hash.

## Result messages:

Each result message is a JSON object that is sent in the following subject:

stan.<model_hash>.<task_hash>.result

The JSON object has the following fields:

- status: the status of the task. This is a string that can be either "progress", "success" or "failure". Success and failure are terminal states, after which we do not expect any more messages.
- message: a message that is passed to the user. This is a string that can be used to provide more information about the status of the task.
- progress: a number between 0 and 1 that indicates the progress of the task. This is a float that can be used to provide more information about the status of the task.
- result: the result of the task. This is a JSON object that is passed to the user. This is only present if the status is "success". 


# Worker cli functionality:

# Client cli functionality:

- List all tasks that are done.
- List all tasks that are in progress.
- List all tasks that have failed.
- List all tasks that are pending.
- Inspection of a specific task.
- 