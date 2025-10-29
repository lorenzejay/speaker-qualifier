import os

from crewai import LLM, TaskOutput
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool, EXASearchTool


from crewai_tools import CrewaiEnterpriseTools
from pydantic import BaseModel


class Score(BaseModel):
    score: int
    reasoning: str
    evidence: list[str]


class SpeakerEvaluationCriteria(BaseModel):
    ai_agents_expertise: Score
    speaking_experience: Score
    thought_leadership: Score
    professional_experience: Score
    workshop_suitability: Score


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

    def speaker_evaluation_guardrail(self, result: TaskOutput) -> tuple[bool, str]:
        from opik.evaluation.metrics import AnswerRelevance

        metric = AnswerRelevance(require_context=False)
        answer_relevance_score = metric.score(result.expected_output, result.raw)
        print(f"Answer relevance score: {answer_relevance_score}")
        if answer_relevance_score.scoring_failed:
            return (
                False,
                "The answer relevance score is too high. The answer_relevance_score  is {answer_relevance_score}",
            )
        else:
            return (
                True,
                "The answer relevance score is low. The answer_relevance_score  is {answer_relevance_score}",
            )

    @task
    def generate_speaker_evaluation_report(self) -> Task:
        return Task(
            config=self.tasks_config["generate_speaker_evaluation_report"],
            markdown=False,
            guardrail=self.speaker_evaluation_guardrail,
            output_pydantic=SpeakerEvaluationCriteria,
        )

    @task
    def send_report_to_slack(self) -> Task:
        return Task(
            config=self.tasks_config["send_report_to_slack"],
            markdown=False,
            guardrail="ensure the message is sent to the slack recipient and the message includes relevant links from the web research for each criteria to support the evidence of experiece.",
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
