from bullmq import Job


def is_final_attempt(job: Job) -> bool:
    attempts = getattr(job, "opts", {}).get("attempts", 1)
    attempts_made = getattr(job, "attemptsMade", 0)
    return attempts_made + 1 >= attempts
