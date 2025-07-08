from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_fundamentals_analyst(llm, toolkit):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        if toolkit.config["online_tools"]:
            tools = [toolkit.get_fundamentals_openai]
        else:
            tools = [
                toolkit.get_finnhub_company_insider_sentiment,
                toolkit.get_finnhub_company_insider_transactions,
                toolkit.get_simfin_balance_sheet,
                toolkit.get_simfin_cashflow,
                toolkit.get_simfin_income_stmt,
            ]

        system_message = (
            "你是一名研究人员，负责分析某家公司过去一周的基本面信息。请撰写一份关于该公司基本面信息的综合报告，内容包括财务文件、公司简介、基本财务状况、财务历史、内部人士情绪和内部交易等，以全面了解该公司的基本面信息，为交易员提供参考。请尽可能包含详细信息，不要简单地说趋势喜忧参半，要提供详细、细致的分析和见解，以帮助交易员做出决策。"
            + " 请在报告末尾附上一个 Markdown 表格，整理报告中的要点，使其有条理且易于阅读。",
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    " 你是一个乐于助人的人工智能助手，与其他助手协同合作。"
                    " 使用所提供的工具来推进问题的解答。"
                    " 如果您无法完全回答，也没关系；另一位拥有不同工具的助手会从您停下的地方继续帮助。执行您能做的部分来推进进展。"
                    # " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    # " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " 如果您或任何其他助手有最终交易建议：**BUY/HOLD/SELL** 或可交付成果，请在回复前加上 最终交易建议：**BUY/HOLD/SELL**，以便团队知道停止工作。"
                    # " You have access to the following tools: {tool_names}.\n{system_message}"
                    " 您可以使用以下工具：{tool_names}。\n{system_message}"
                    # "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                    " 供您参考，当前日期是 {current_date}。我们要研究的公司是 {ticker}"
                    " 所有答复请使用简体中文"
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
