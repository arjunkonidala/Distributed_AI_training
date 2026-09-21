# mcp/server.py

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CareFlow")


@mcp.tool()
def update_patient_record(
    patient_id: str,
    triage_level: str
) -> str:

    # Replace with actual DB operation
    print(
        f"Updating patient {patient_id}: "
        f"{triage_level}"
    )

    return "Patient record updated successfully"


@mcp.tool()
def send_notification(
    channel: str,
    message: str
) -> str:

    # Replace with Slack/email/etc.
    print(
        f"Sending notification to {channel}: "
        f"{message}"
    )

    return "Notification sent"


if __name__ == "__main__":
    mcp.run()
