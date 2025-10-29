import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool, EXASearchTool


from crewai_tools import CrewaiEnterpriseTools


@CrewBase
class OdscAiAgentsSpeakerEvaluationPipelineCrew:
    """OdscAiAgentsSpeakerEvaluationPipeline crew"""

    @agent
    def web_research_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["web_research_specialist"],
            tools=[ScrapeWebsiteTool(), EXASearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4.1-mini",
                temperature=0.7,
            ),
        )

    @agent
    def odsc_speaker_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["odsc_speaker_evaluator"],
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4.1-mini",
                temperature=0.7,
            ),
        )

    @agent
    def report_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["report_generator"],
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4.1-mini",
                temperature=0.7,
            ),
        )

    @agent
    def slack_notifier(self) -> Agent:
        enterprise_actions_tool = CrewaiEnterpriseTools(
            enterprise_token=os.getenv("CREWAI_ENTERPRISE_TOOLS_TOKEN_LORENZE"),
            actions_list=[
                "slack_get_user_by_email",
                "slack_send_message",
                "slack_get_users_by_name",
            ],
        )

        return Agent(
            config=self.agents_config["slack_notifier"],
            tools=[*enterprise_actions_tool],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4.1-mini",
                temperature=0.7,
            ),
        )

    @task
    def comprehensive_web_research(self) -> Task:
        return Task(
            config=self.tasks_config["comprehensive_web_research"],
            markdown=False,
        )

    @task
    def evaluate_odsc_speaker_potential(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_odsc_speaker_potential"],
            markdown=False,
        )

    @task
    def generate_speaker_evaluation_report(self) -> Task:
        return Task(
            config=self.tasks_config["generate_speaker_evaluation_report"],
            markdown=False,
        )

    @task
    def send_report_to_slack(self) -> Task:
        return Task(
            config=self.tasks_config["send_report_to_slack"],
            markdown=False,
            guardrail="ensure the message is sent to the slack recipient",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the OdscAiAgentsSpeakerEvaluationPipeline crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            tracing=True,
        )
