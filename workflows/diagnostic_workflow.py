"""
Improved diagnostic workflow with DTC code normalization + language control + comprehensive logging
"""

import logging
import asyncio
import re
import json
import re
import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

from tools.RAG_tools import (
    is_vehicle_related, extract_vehicle_model, search_vehicle_documents,
    grade_document_relevance, search_web_for_vehicle_info, search_youtube_videos,
    format_diagnostic_results, llm as tools_llm, enhance_question_with_vehicle
)

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diagnostic_workflow.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

TOOLS = [
    is_vehicle_related, extract_vehicle_model, search_vehicle_documents,
    grade_document_relevance, search_web_for_vehicle_info, search_youtube_videos,
    format_diagnostic_results, enhance_question_with_vehicle
]

class SessionManager:
    """Manages session information and conversation IDs."""
    
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        self.conversation_counter = 0
        self.session_dir = self._create_session_directory()
        self._save_session_info()
        
        logger.info(f"🚀 NEW SESSION STARTED: {self.session_id}")
        logger.info(f"📁 Session logs directory: {self.session_dir}")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        return f"session_{timestamp}_{short_uuid}"
    
    def _create_session_directory(self) -> str:
        """Create a directory for this session's logs."""
        session_dir = f"conversation_logs/{self.session_id}"
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def _save_session_info(self):
        """Save session metadata."""
        session_info = {
            "session_id": self.session_id,
            "start_time": self.session_start.isoformat(),
            "total_conversations": 0,
            "last_activity": self.session_start.isoformat(),
            "status": "active"
        }
        
        session_file = f"{self.session_dir}/session_info.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session info: {e}")
    
    def get_next_conversation_id(self) -> str:
        """Get the next sequential conversation ID."""
        self.conversation_counter += 1
        conversation_id = f"conv_{self.conversation_counter:04d}"
        return conversation_id
    
    def update_session_activity(self):
        """Update last activity timestamp."""
        try:
            session_file = f"{self.session_dir}/session_info.json"
            if os.path.exists(session_file):
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_info = json.load(f)
                
                session_info["last_activity"] = datetime.now().isoformat()
                session_info["total_conversations"] = self.conversation_counter
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to update session activity: {e}")

class ConversationLogger:
    """Enhanced logging system for tracking all tool calls and responses with session management."""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.conversations: Dict[str, Dict] = {}
        
    def log_conversation_start(self, thread_id: str, query: str, language: str):
        """Log the start of a new conversation with sequential ID."""
        conversation_id = self.session_manager.get_next_conversation_id()
        full_conversation_id = f"{self.session_manager.session_id}_{conversation_id}"
        
        conversation_data = {
            "session_id": self.session_manager.session_id,
            "conversation_id": conversation_id,
            "full_id": full_conversation_id,
            "thread_id": thread_id,
            "conversation_number": self.session_manager.conversation_counter,
            "start_time": datetime.now().isoformat(),
            "language": language,
            "initial_query": query,
            "tool_calls": [],
            "responses": [],
            "final_response": None,
            "errors": [],
            "metadata": {
                "total_tools_called": 0,
                "execution_time_seconds": None,
                "success": False,
                "session_info": {
                    "session_start": self.session_manager.session_start.isoformat(),
                    "conversation_in_session": self.session_manager.conversation_counter
                }
            }
        }
        
        self.conversations[full_conversation_id] = conversation_data
        
        logger.info(f"🎯 NEW CONVERSATION STARTED")
        logger.info(f"   Session: {self.session_manager.session_id}")
        logger.info(f"   Conversation: #{self.session_manager.conversation_counter:04d} ({conversation_id})")
        logger.info(f"   Full ID: {full_conversation_id}")
        logger.info(f"   Thread: {thread_id}")
        logger.info(f"   Language: {language}")
        logger.info(f"   Query: {query}")
        
        return full_conversation_id
    
    def log_tool_call(self, conversation_id: str, tool_name: str, inputs: Dict, step_number: int):
        """Log a tool call with inputs."""
        if conversation_id not in self.conversations:
            return
            
        tool_call_log = {
            "step": step_number,
            "tool_name": tool_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": inputs,
            "output": None,
            "execution_time_ms": None,
            "error": None
        }
        
        self.conversations[conversation_id]["tool_calls"].append(tool_call_log)
        
        conv_num = self.conversations[conversation_id]["conversation_number"]
        logger.info(f"🔧 TOOL CALL #{step_number} [Conv #{conv_num:04d}]")
        logger.info(f"   Tool: {tool_name}")
        logger.info(f"   Inputs: {json.dumps(inputs, indent=2)}")
        
    def log_tool_response(self, conversation_id: str, tool_name: str, output: Any, execution_time_ms: float, error: Optional[str] = None):
        """Log tool response and execution time."""
        if conversation_id not in self.conversations:
            return
            
        # Find the latest tool call for this tool
        for tool_call in reversed(self.conversations[conversation_id]["tool_calls"]):
            if tool_call["tool_name"] == tool_name and tool_call["output"] is None:
                tool_call["output"] = output
                tool_call["execution_time_ms"] = execution_time_ms
                tool_call["error"] = error
                break
        
        self.conversations[conversation_id]["metadata"]["total_tools_called"] += 1
        
        conv_num = self.conversations[conversation_id]["conversation_number"]
        logger.info(f"✅ TOOL RESPONSE [Conv #{conv_num:04d}]")
        logger.info(f"   Tool: {tool_name}")
        logger.info(f"   Execution Time: {execution_time_ms:.2f}ms")
        if error:
            logger.error(f"   Error: {error}")
        else:
            # Truncate output for logging but keep full output in data
            output_preview = str(output)[:500] + "..." if len(str(output)) > 500 else str(output)
            logger.info(f"   Output Preview: {output_preview}")
    
    def log_conversation_end(self, conversation_id: str, final_response: str, success: bool = True):
        """Log the end of a conversation."""
        if conversation_id not in self.conversations:
            return
            
        conversation_data = self.conversations[conversation_id]
        start_time = datetime.fromisoformat(conversation_data["start_time"])
        execution_time = (datetime.now() - start_time).total_seconds()
        
        conversation_data["final_response"] = final_response
        conversation_data["metadata"]["execution_time_seconds"] = execution_time
        conversation_data["metadata"]["success"] = success
        conversation_data["end_time"] = datetime.now().isoformat()
        
        conv_num = conversation_data["conversation_number"]
        conv_id = conversation_data["conversation_id"]
        
        logger.info(f"🏁 CONVERSATION ENDED")
        logger.info(f"   Session: {self.session_manager.session_id}")
        logger.info(f"   Conversation: #{conv_num:04d} ({conv_id})")
        logger.info(f"   Success: {success}")
        logger.info(f"   Total Execution Time: {execution_time:.2f}s")
        logger.info(f"   Tools Called: {conversation_data['metadata']['total_tools_called']}")
        logger.info(f"   Final Response Length: {len(final_response)} chars")
        
        # Save detailed log to file
        self._save_conversation_to_file(conversation_id)
        
        # Update session activity
        self.session_manager.update_session_activity()
    
    def log_error(self, conversation_id: str, error: str, context: str = ""):
        """Log an error during conversation."""
        if conversation_id not in self.conversations:
            return
            
        error_log = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context
        }
        
        self.conversations[conversation_id]["errors"].append(error_log)
        conv_num = self.conversations[conversation_id]["conversation_number"]
        logger.error(f"❌ ERROR [Conv #{conv_num:04d}]: {error} | Context: {context}")
    
    def _save_conversation_to_file(self, conversation_id: str):
        """Save individual conversation log to file in session directory."""
        if conversation_id not in self.conversations:
            return
        
        conversation_data = self.conversations[conversation_id]
        conv_id = conversation_data["conversation_id"]
        
        # Save in session directory with descriptive filename
        filename = f"{self.session_manager.session_dir}/{conv_id}_{conversation_data['start_time'][:19].replace(':', '-')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Conversation log saved: {filename}")
            
            # Also save a summary file for quick overview
            self._save_session_summary()
            
        except Exception as e:
            logger.error(f"Failed to save conversation log: {e}")
    
    def _save_session_summary(self):
        """Save a summary of all conversations in this session."""
        summary = {
            "session_id": self.session_manager.session_id,
            "session_start": self.session_manager.session_start.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_conversations": len(self.conversations),
            "conversations": []
        }
        
        for conv_id, conv_data in self.conversations.items():
            summary["conversations"].append({
                "conversation_number": conv_data["conversation_number"],
                "conversation_id": conv_data["conversation_id"],
                "start_time": conv_data["start_time"],
                "end_time": conv_data.get("end_time", "In Progress"),
                "query": conv_data["initial_query"],
                "language": conv_data["language"],
                "success": conv_data["metadata"]["success"],
                "tools_called": conv_data["metadata"]["total_tools_called"],
                "execution_time": conv_data["metadata"]["execution_time_seconds"],
                "response_length": len(conv_data["final_response"]) if conv_data["final_response"] else 0,
                "errors_count": len(conv_data["errors"])
            })
        
        # Sort by conversation number for easy reading
        summary["conversations"].sort(key=lambda x: x["conversation_number"])
        
        summary_file = f"{self.session_manager.session_dir}/session_summary.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session summary: {e}")
    
    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """Get a summary of a specific conversation."""
        if conversation_id not in self.conversations:
            return {}
        
        conv = self.conversations[conversation_id]
        return {
            "session_id": self.session_manager.session_id,
            "conversation_number": conv["conversation_number"],
            "conversation_id": conv["conversation_id"],
            "full_id": conversation_id,
            "query": conv["initial_query"],
            "language": conv["language"],
            "tools_used": [tc["tool_name"] for tc in conv["tool_calls"]],
            "total_execution_time": conv["metadata"]["execution_time_seconds"],
            "success": conv["metadata"]["success"],
            "response_length": len(conv["final_response"]) if conv["final_response"] else 0
        }
    
    def get_session_info(self) -> Dict:
        """Get current session information."""
        return {
            "session_id": self.session_manager.session_id,
            "session_start": self.session_manager.session_start.isoformat(),
            "conversations_count": len(self.conversations),
            "last_conversation_number": self.session_manager.conversation_counter,
            "session_directory": self.session_manager.session_dir
        }

# Global logger instance
conversation_logger = ConversationLogger()

class AsyncDiagnosticAgent:
    """
    Async-first diagnostic agent with DTC code normalization and target-language control.
    Enhanced with comprehensive session-based logging.
    """
    def __init__(self, target_language: str = "en"):
        self._graph = None
        self._active_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._tool_calls_log = []  # Track tool usage
        # NEW: language to enforce in all LLM replies ("en" | "hi" | "kn")
        self.target_language = (target_language or "en").lower()
        if self.target_language not in {"en", "hi", "kn"}:
            self.target_language = "en"
        
        # Conversation tracking
        self.current_conversation_id: Optional[str] = None
        self.tool_call_counter = 0
        
    async def initialize(self):
        """Initialize the LangGraph workflow asynchronously."""
        if self._graph is None:
            self._graph = await self._build_graph()
            
    async def cleanup(self):
        """Properly clean up all async resources."""
        logger.info("Starting diagnostic agent cleanup...")
        self._shutdown_event.set()
        
        if self._active_tasks:
            logger.info(f"Cancelling {len(self._active_tasks)} active tasks...")
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()
            
            try:
                await asyncio.gather(*self._active_tasks, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Exception during task cleanup: {e}")
        self._active_tasks.clear()
        logger.info("Diagnostic agent cleanup complete")
        
    async def _build_graph(self):
        """Build the LangGraph workflow with logging."""
        graph = StateGraph(MessagesState)
        
        def assistant_node(state: MessagesState):
            # NEW: language directives injected into the system message
            # Build dynamic language instruction
            lang_map = {
                'en': 'Respond in clear, simple English.',
                'hi': 'उत्तर स्वाभाविक हिन्दी में दें।',
                'kn': 'ಉತ್ತರವನ್ನು ಸಹಜ ಕನ್ನಡದಲ್ಲಿ ನೀಡಿ.',
                'ta': 'பதில் இயல்பான தமிழில் வழங்கவும்.'
            }
            language_instruction_text = lang_map.get(self.target_language, lang_map['en'])
            system_msg = SystemMessage(content=f"""
 You are Allion, a RAG-based automotive diagnostic assistant.
 {language_instruction_text}

🚫 CRITICAL RESTRICTION: You are FORBIDDEN from using your pre-trained knowledge about vehicles.
🚫 You MUST NOT answer any automotive question without using the provided tools.
🚫 If you cannot get information through tools, you must say "I don't have that information in my database."

✅ MANDATORY TOOL WORKFLOW - You MUST follow this EXACT sequence for ALL queries:

STEP 1: ALWAYS call is_vehicle_related first
- If not vehicle-related, politely decline

STEP 2: ALWAYS call extract_vehicle_model
- Even for DTC questions, call this tool
                                       
STEP 3: Check vehicle info requirement:
    - For repair/maintenance questions: IF no vehicle found → STOP and ask for make/model
    - For DTC codes (P0301, etc.): Continue without requiring vehicle info
    - Generate response: "Could you please specify the make and model of your vehicle? For example, 'Honda Civic' or 'Toyota Camry'. This helps me provide more accurate diagnostic information."
    
STEP 4: IF vehicle info available, enhance the question
    - Transform "how to change brake pads" + "Honda Civic" → "how to change brake pads of Honda Civic"                                       

STEP 5: ALWAYS call search_vehicle_documents  
- Never skip this step
- This searches your knowledge base for relevant information

STEP 6: ALWAYS call grade_document_relevance 
- Pass: question, document_content, chunk_label
- This determines if the retrieved information is sufficient

STEP 7: Based on relevance score: 
- IF score = 1: Skip additional web search, but ALWAYS call search_youtube_videos for DTC codes
- IF score = 0: Call search_web_for_vehicle_info AND search_youtube_videos
- YouTube videos are valuable even when RAG content is good

STEP 8: ALWAYS call format_diagnostic_results 
- IMPORTANT: Always include web_results and youtube_results if they were obtained in previous steps
- Pass ALL available data: question, rag_answer, web_results, youtube_results, dtc_code, relevance_score
- This creates your final response with ALL information (both voice_output and text_output)

🎯 CRITICAL VOICE/TEXT SEPARATION (For System Architecture):
- format_diagnostic_results returns BOTH voice_output (concise) and text_output (detailed)
- The system AUTOMATICALLY handles separation:
  * voice_output → sent to TTS via LLM stream (for speaking)
  * text_output → stored at agent level for separate frontend retrieval (for diagnostic panel)
- Your format_diagnostic_results tool output is split automatically; you create complete structured response

🚫 FORBIDDEN BEHAVIORS:
- Do NOT answer questions directly without using tools
- Do NOT use your knowledge about P0301, engine codes, or any automotive topics
- Do NOT provide diagnostic advice without retrieving it through tools
- Do NOT skip any steps in the workflow
- Do NOT omit web_results or youtube_results from format_diagnostic_results if they exist

✅ REQUIRED RESPONSES:
- If tools return no information: "I don't have specific information about this in my diagnostic database."
- If you're tempted to answer from memory: STOP and use tools instead
- Always base your response ONLY on tool results
- Always include ALL available web sources and YouTube videos in the final formatting

REMEMBER: You are a RAG assistant, not a general automotive expert. Your knowledge comes ONLY from the tools.
 The system architecture ensures voice and text are separated after your response is generated.
""")
            
            messages = [system_msg] + state.get("messages", [])
            tools_llm_with_tools = tools_llm.bind_tools(TOOLS)
            
            # Log the assistant call
            if self.current_conversation_id:
                conv_data = conversation_logger.conversations.get(self.current_conversation_id, {})
                conv_num = conv_data.get("conversation_number", "Unknown")
                logger.info(f"🤖 ASSISTANT NODE called [Conv #{conv_num:04d}]")
            
            response = tools_llm_with_tools.invoke(messages)
            
            # Log tool calls if present
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    self.tool_call_counter += 1
                    if self.current_conversation_id:
                        conversation_logger.log_tool_call(
                            self.current_conversation_id,
                            tool_call['name'],
                            tool_call['args'],
                            self.tool_call_counter
                        )
            
            return {"messages": [response]}
        
        def logged_tool_node(state: MessagesState):
            """Wrapped tool node with logging."""
            start_time = datetime.now()
            
            # Get the tool calls from the last message
            messages = state.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call['name']
                        # Log tool execution start
                        conv_data = conversation_logger.conversations.get(self.current_conversation_id, {})
                        conv_num = conv_data.get("conversation_number", "Unknown")
                        logger.info(f"🛠️ EXECUTING TOOL: {tool_name} [Conv #{conv_num:04d}]")
            
            # Execute the actual tool node
            try:
                # FIX: Create ToolNode instance and invoke properly
                tool_node = ToolNode(TOOLS)
                result = tool_node.invoke(state)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log successful execution
                if self.current_conversation_id and messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call['name']
                            # Get the tool output from result
                            tool_output = "Tool executed successfully"
                            if 'messages' in result:
                                for msg in result['messages']:
                                    if hasattr(msg, 'content'):
                                        tool_output = msg.content
                                        break
                            
                            conversation_logger.log_tool_response(
                                self.current_conversation_id,
                                tool_name,
                                tool_output,
                                execution_time
                            )
                
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(f"❌ TOOL EXECUTION ERROR: {e}")
                
                if self.current_conversation_id and messages:
                    last_message = messages[-1]
                    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call['name']
                            conversation_logger.log_tool_response(
                                self.current_conversation_id,
                                tool_name,
                                None,
                                execution_time,
                                str(e)
                            )
                            conversation_logger.log_error(
                                self.current_conversation_id,
                                str(e),
                                f"Tool execution failed: {tool_name}"
                            )
                raise
            
        graph.add_node("assistant", assistant_node)
        graph.add_node("tools", logged_tool_node)
        graph.set_entry_point("assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
        
        return graph.compile(checkpointer=MemorySaver())
        
    async def _handle_smalltalk_async(self, message: str) -> str | None:
        """Handle common small talk or reject non-vehicle chit-chat."""
        msg = message.strip().lower()
        
        # Add debug logging
        logger.info(f"🔍 Smalltalk check - Original: '{message}' | Cleaned: '{msg}'")

        # ✅ More flexible greeting check
        if any(word in msg for word in ["hi", "hello", "hey"]):
            logger.info(f"✅ Greeting detected in: '{msg}'")
            if self.target_language == "hi":
                return "नमस्ते, मैं एलियन हूँ 👋 आपकी गाड़ी में किस बात में मदद करूँ?"
            if self.target_language == "kn":
                return "ಹಲೋ, ನಾನು ಅಲಿಯನ್ 👋 ನಿಮ್ಮ ವಾಹನದ ಬಗ್ಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
            return "Hi, I am Allion 👋 How can I assist you with your vehicle today?"

        if "how are you" in msg:
            logger.info(f"✅ 'How are you' detected in: '{msg}'")
            if self.target_language == "hi":
                return "मैं ठीक हूँ, धन्यवाद! आपकी गाड़ी से जुड़ी किस बात में मदद करूँ?"
            if self.target_language == "kn":
                return "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ, ಧನ್ಯವಾದಗಳು! ನಿಮ್ಮ ವಾಹನದ ಬಗ್ಗೆ ನಾನು ಏನು ಸಹಾಯ ಮಾಡಲಿ?"
            return "I'm doing fine, thank you! How can I assist you with your vehicle?"

        if any(word in msg for word in ["bye", "goodbye", "stop", "exit", "quit"]):
            logger.info(f"✅ Farewell detected in: '{msg}'")
            if self.target_language == "hi":
                return "ठीक है, आपका दिन शुभ हो! 🚗💨"
            if self.target_language == "kn":
                return "ಸರಿ, ನಿಮಗೆ ಶುಭ ದಿನ! 🚗💨"
            return "Sure, have a nice day! 🚗💨"

        if msg in ["no", "nope", "nah"]:
            logger.info(f"✅ Negative response detected: '{msg}'")
            if self.target_language == "hi":
                return "धन्यवाद! आपका दिन शुभ हो। 👋"
            if self.target_language == "kn":
                return "ಧನ್ಯವಾದಗಳು! ನಿಮಗೆ ಶುಭ ದಿನ. 👋"
            return "Thank you! Have a nice day. 👋"

        if any(word in msg for word in ["thanks", "thank you", "thx"]):
            logger.info(f"✅ Thanks detected in: '{msg}'")
            if self.target_language == "hi":
                return "आपका स्वागत है! क्या आपको अपनी गाड़ी के बारे में और मदद चाहिए?"
            if self.target_language == "kn":
                return "ಸ್ವಾಗತ! ನಿಮ್ಮ ವಾಹನದ ಬಗ್ಗೆ ಇನ್ನಷ್ಟು ಸಹಾಯ ಬೇಕೆ?"
            return "You're welcome! Do you need any more help with your vehicle?"
        
        try:
            from tools.RAG_tools import is_vehicle_related
            
            result = is_vehicle_related.invoke({"question": message})
            is_vehicle = result.get("is_vehicle_related", False)
            
            if not is_vehicle:
                if self.target_language == "hi":
                    return "मैं वाहन निदान में विशेषज्ञ हूँ 🚗। कृपया कार से संबंधित प्रश्न पूछें।"
                if self.target_language == "kn":
                    return "ನಾನು ವಾಹನ ಡಯಾಗ್ನೋಸ್ಟಿಕ್ಸ್‌ನಲ್ಲಿ ಪರಿಣತಿ ಹೊಂದಿದ್ದೇನೆ 🚗. ದಯವಿಟ್ಟು ಕಾರಿಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆ ಕೇಳಿ."
                return "I specialize in vehicle diagnostics 🚗. Could you please ask me something related to your car?"
            
        except Exception as e:
            logger.warning(f"Error checking if message is vehicle-related: {e}")

        # ✅ If it's vehicle-related → continue with RAG
        logger.info(f"✅ Vehicle-related detected: '{msg}' - continuing with RAG")
        return None

    async def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Process a chat message with comprehensive logging.
        """
        if self._shutdown_event.is_set():
            return {
                "hi": "एजेंट बंद हो रहा है, अभी अनुरोध प्रोसेस नहीं कर सकता।",
                "kn": "ಏಜೆಂಟ್ ಮುಚ್ಚಲಾಗುತ್ತಿದೆ, ಈಗ ವಿನಂತಿಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ.",
                "en": "Agent is shutting down, cannot process request.",
            }[self.target_language]

        # Start conversation logging
        self.current_conversation_id = conversation_logger.log_conversation_start(
            thread_id, message, self.target_language
        )
        self.tool_call_counter = 0

        try:
            # ✅ Step 1: Smalltalk / non-vehicle handling (now async)
            smalltalk_response = await self._handle_smalltalk_async(message)
            if smalltalk_response:
                conversation_logger.log_conversation_end(
                    self.current_conversation_id, smalltalk_response, True
                )
                return smalltalk_response

            # ✅ Step 2: Continue with diagnostic RAG workflow
            await self.initialize()
            task = asyncio.create_task(self._process_message(message, thread_id))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

            result = await task
            
            # Log successful completion
            conversation_logger.log_conversation_end(
                self.current_conversation_id, result, True
            )
            
            return result
            
        except asyncio.CancelledError:
            logger.info("Chat task was cancelled")
            error_msg = {
                "hi": "अनुरोध रद्द कर दिया गया।",
                "kn": "ವಿನಂತಿಯನ್ನು ರದ್ದುಪಡಿಸಲಾಗಿದೆ.",
                "en": "Request was cancelled",
            }[self.target_language]
            
            if self.current_conversation_id:
                conversation_logger.log_error(
                    self.current_conversation_id, "Request cancelled", "AsyncCancelledError"
                )
                conversation_logger.log_conversation_end(
                    self.current_conversation_id, error_msg, False
                )
            
            return error_msg
            
        except Exception as e:
            logger.exception("Error in chat processing")
            error_msg = {
                "hi": f"त्रुटि: {e}",
                "kn": f"ದೋಷ: {e}",
                "en": f"Error processing your question: {e}",
            }[self.target_language]
            
            if self.current_conversation_id:
                conversation_logger.log_error(
                    self.current_conversation_id, str(e), "Chat processing error"
                )
                conversation_logger.log_conversation_end(
                    self.current_conversation_id, error_msg, False
                )
            
            return error_msg

    async def _process_message(self, message: str, thread_id: str) -> str:
        """Internal message processing with tool validation."""
        try:
            # Reset tool call tracking
            self._tool_calls_log = []
            
            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Check if tools were actually called
            messages = result.get("messages", [])
            tool_calls_made = any(
                hasattr(msg, 'tool_calls') and msg.tool_calls 
                for msg in messages
            )
            
            if not tool_calls_made:
                logger.warning(f"⚠️ LLM tried to answer without using tools: {message}")
                if self.current_conversation_id:
                    conversation_logger.log_error(
                        self.current_conversation_id, 
                        "LLM attempted to answer without tools", 
                        "Tool validation failed"
                    )
                return {
                    "hi": "क्षमा कीजिए, यह वाहन से संबंधित नहीं लग रहा। कृपया कार से जुड़ा प्रश्न पूछें।",
                    "kn": "ಕ್ಷಮಿಸಿ, ಇದು ವಾಹನಕ್ಕೆ ಸಂಬಂಧಿಸಿದಂತೆ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಕಾರಿಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಶ್ನೆ ಕೇಳಿ.",
                    "en": "Sorry, I'm not sure if this is related to a vehicle. Could you please ask me specifically about vehicle-related questions?",
                }[self.target_language]
            
            # Extract assistant response and check for structured format
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content and not isinstance(msg, HumanMessage):
                    # Check if this is a tool message with structured response
                    if hasattr(msg, 'name') and msg.name == 'format_diagnostic_results':
                        # This is the response from format_diagnostic_results tool
                        try:
                            tool_result = json.loads(msg.content)
                            if 'formatted_response' in tool_result:
                                formatted_response = tool_result['formatted_response']
                                
                                # CRITICAL: Extract and store text_output internally
                                # Only return voice_output for final_response logging
                                voice_output = formatted_response.get('voice_output', '')
                                text_output = formatted_response.get('text_output', {})
                                
                                # Store text internally for LiveKit stream
                                self._diagnostic_text_payload = json.dumps(text_output)
                                logger.info(f"📊 Extracted text_output, stored internally. Voice: {len(voice_output)} chars, Text payload: {len(self._diagnostic_text_payload)} chars")
                                
                                # Return ONLY voice_output for logging and LLM stream
                                return voice_output
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse structured response: {e}")
                            # Fall back to plain text
                            return msg.content
                    
                    # Regular assistant message (non-tool)
                    elif not hasattr(msg, 'name'):
                        # Validate the response doesn't contain generic automotive knowledge
                        if self._contains_generic_knowledge(msg.content):
                            logger.warning("⚠️ Response contains generic knowledge, not tool output")
                            if self.current_conversation_id:
                                conversation_logger.log_error(
                                    self.current_conversation_id,
                                    "Response contains generic knowledge instead of RAG output",
                                    "Knowledge validation failed"
                                )
                            return {
                                "hi": "ध्यान दें: यह उत्तर डायग्नोस्टिक डेटाबेस के बजाय सामान्य ज्ञान पर आधारित है।",
                                "kn": "ಗಮನಿಸಿ: ಈ ಉತ್ತರವು ಡಯಗ್ನೋಸ್ಟಿಕ್ ಡೇಟಾಬೇಸ್‌ಗಿಂತ ಸಾಮಾನ್ಯ ಜ್ಞಾನ ಆಧಾರಿತವಾಗಿದೆ.",
                                "en": "I should let you know: I'm answering this using my pre-trained/existing knowledge, not from the diagnostic database.",
                            }[self.target_language]
                        return msg.content
            
            return {
                "hi": "क्षमा करें, मैं आपके अनुरोध को प्रोसेस नहीं कर सका।",
                "kn": "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ವಿನಂತಿಯನ್ನು ಪ್ರಕ್ರಿಯೆಗೊಳಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ.",
                "en": "I'm sorry, I couldn't process your request.",
            }[self.target_language]
        except Exception as e:
            logger.exception("Error in _process_message")
            if self.current_conversation_id:
                conversation_logger.log_error(
                    self.current_conversation_id, str(e), "_process_message error"
                )
            raise

    def _contains_generic_knowledge(self, response: str) -> bool:
        """Check if response contains generic automotive knowledge instead of RAG results."""
        # Look for phrases that suggest pre-trained knowledge use
        generic_phrases = [
            "typically", "usually", "commonly", "in general", "most vehicles",
            "based on my knowledge", "automotive industry standard",
            "from my training", "general automotive practice"
        ]
        
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in generic_phrases)

    @asynccontextmanager
    async def session(self):
        """Context manager for proper resource management."""
        try:
            await self.initialize()
            yield self
        finally:
            await self.cleanup()

    def get_conversation_logs(self) -> Dict:
        """Get all conversation logs."""
        return conversation_logger.conversations

    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """Get summary of a specific conversation."""
        return conversation_logger.get_conversation_summary(conversation_id)

# Factory function for configs
async def create_diagnostic_agent(target_language: str = "en"):
    """Create and initialize a diagnostic agent with language control."""
    agent = AsyncDiagnosticAgent(target_language=target_language)
    await agent.initialize()
    return agent

# LiveKit adapter glue
from livekit.agents.llm import LLM, LLMStream, ChatContext, ChatMessage, FunctionTool, RawFunctionTool, ToolChoice, ChatChunk, ChoiceDelta
from livekit.agents import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from typing import Any

class DiagnosticLLMAdapter(LLM):
    """
    Adapter that implements LiveKit LLM interface using async agent internally,
    with enforced target language and WebSocket support for diagnostic text delivery.
    """
    def __init__(self, target_language: str = "en"):
        super().__init__()
        self._agent: Optional[AsyncDiagnosticAgent] = None
        self._loop = None
        self.target_language = (target_language or "en").lower()
        if self.target_language not in {"en", "hi", "kn"}:
            self.target_language = "en"
        
        # WebSocket reference for sending diagnostic text
        self._websocket = None
        self._ws_ready = asyncio.Event()

    def set_websocket(self, websocket) -> None:
        """
        Set the WebSocket connection for sending diagnostic text.
        Called by LiveKit worker when connection is established.
        
        Args:
            websocket: The WebSocket connection object
        """
        self._websocket = websocket
        self._ws_ready.set()
        logger.info("✅ WebSocket connection set for diagnostic text delivery")

    async def send_diagnostic_text_via_websocket(self, text_payload: str) -> bool:
        """
        Send diagnostic text payload through WebSocket to frontend.
        
        Args:
            text_payload: JSON string containing diagnostic text
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self._websocket:
            logger.warning("⚠️ WebSocket not available for diagnostic text delivery")
            return False
        
        try:
            # Wait a bit for WebSocket to be fully ready
            await asyncio.wait_for(self._ws_ready.wait(), timeout=1.0)
            
            message = json.dumps({
                "type": "diagnostic:text",
                "payload": text_payload,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send via WebSocket
            await self._websocket.send(message)
            logger.info(f"📤 Diagnostic text sent via WebSocket: {len(text_payload)} chars")
            return True
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ WebSocket not ready in time for diagnostic text delivery")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send diagnostic text via WebSocket: {e}")
            return False

    def _ensure_agent(self):
        """Ensure agent is created in the current event loop."""
        current_loop = asyncio.get_event_loop()
        if self._agent is None or self._loop != current_loop:
            self._loop = current_loop
            self._agent = AsyncDiagnosticAgent(target_language=self.target_language)
            # Schedule initialization
            self._loop.create_task(self._agent.initialize())

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools = None,
        conn_options = None,
        parallel_tool_calls = None,
        tool_choice = None,
        extra_kwargs = None,
    ) -> LLMStream:
        """
        Implement the LiveKit LLM chat interface with DTC normalization.
        """
        self._ensure_agent()
        
        # Extract the latest user message from the chat context
        messages = chat_ctx.items
        user_message = ""
        
        if messages:
            latest_message = messages[-1]
            if hasattr(latest_message, 'content') and latest_message.content:
                if isinstance(latest_message.content, list) and latest_message.content:
                    user_message = str(latest_message.content[0])
                else:
                    user_message = str(latest_message.content)

        return DiagnosticLLMStream(
            llm=self, 
            agent=self._agent, 
            message=user_message,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options or DEFAULT_API_CONNECT_OPTIONS,
        )

    def get_diagnostic_text_payload(self) -> Optional[str]:
        """
        Retrieve the diagnostic text payload stored in agent.
        Frontend calls this SEPARATELY (not via LLM stream) to get diagnostic report.
        This ensures text never goes to TTS.
        """
        if self._agent:
            payload = getattr(self._agent, '_diagnostic_text_payload', None)
            if payload:
                logger.info(f"📋 Frontend retrieving diagnostic text: {len(payload)} chars")
            return payload
        return None

    async def aclose(self):
        """Clean up resources."""
        if self._agent:
            await self._agent.cleanup()

class DiagnosticLLMStream(LLMStream):
    """
    Stream implementation for the DiagnosticLLMAdapter.
    """
    def __init__(self, llm: DiagnosticLLMAdapter, agent: AsyncDiagnosticAgent, message: str, chat_ctx, tools=None, conn_options=None):
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._agent = agent
        self._message = message
        self._response_future = None
        self._response_sent = False

    async def _run(self):
        """Required abstract method implementation for LLMStream."""
        pass

    async def __anext__(self) -> ChatChunk:
        """Get the next chunk from the stream.
        
        CRITICAL: Only emit voice chunk for TTS consumption.
        Text payload is stored at agent level and sent via separate frontend channel.
        
        This prevents TTS from reading diagnostic JSON.
        """
        # Only emit voice once
        if getattr(self, '_voice_sent', False):
            raise StopAsyncIteration

        if self._response_future is None:
            self._response_future = asyncio.create_task(self._get_response())

        try:
            # ONLY chunk: voice summary (safe for TTS)
            # Text is handled separately at frontend level via agent state
            voice = await self._response_future  # returns voice only
            self._voice_sent = True
            logger.info(f"📤 Emitting voice chunk to TTS: {len(voice)} chars")
            return ChatChunk(
                id="diagnostic_voice",
                delta=ChoiceDelta(
                    role="assistant",
                    content=voice
                )
            )
        except StopAsyncIteration:
            # Re-raise StopAsyncIteration without logging - it's normal control flow
            raise
        except Exception as e:
            logger.exception("Error in DiagnosticLLMStream.__anext__")
            self._voice_sent = True
            return ChatChunk(
                id="diagnostic_error",
                delta=ChoiceDelta(
                    role="assistant",
                    content=f"Error: {e}"
                )
            )

    async def _get_response(self) -> str:
        """Get the response from the diagnostic agent and handle structured format.
        
        Returns ONLY voice_output string. Stores text_output separately for second chunk.
        TTS will only receive the returned voice string from the first chunk.
        """
        try:
            response = await self._agent.chat(self._message)
            
            # Check if response is structured JSON
            try:
                parsed_response = json.loads(response)
                
                # Check for the new structured format with nested formatted_response
                if (isinstance(parsed_response, dict) and 
                    'formatted_response' in parsed_response and
                    isinstance(parsed_response['formatted_response'], dict)):
                    
                    formatted_data = parsed_response['formatted_response']
                    if 'voice_output' in formatted_data and 'text_output' in formatted_data:
                        # This is the new structured response format
                        logger.info("📱 Received structured response from diagnostic agent (new format)")
                        
                        voice_output = formatted_data.get('voice_output', '')
                        text_output = formatted_data.get('text_output', {})

                        # Defensive sanitization right before serialization to prevent TTS pollution.
                        if isinstance(text_output, dict):
                            # Ensure expected keys only
                            allowed_keys = {"content", "web_sources", "youtube_videos", "has_external_sources"}
                            for k in list(text_output.keys()):
                                if k not in allowed_keys:
                                    text_output.pop(k, None)
                            # Strip any embedded HTML blobs inside youtube_videos entries
                            vids = text_output.get('youtube_videos', []) or []
                            cleaned_vids = []
                            import re as _re
                            for v in vids:
                                if not isinstance(v, dict):
                                    continue
                                raw_url = v.get('url', '')
                                # Extract first clean YouTube URL
                                m = _re.search(r'https?://(?:www\.)?(?:youtube\.com/watch\?v=[A-Za-z0-9_-]+|youtu\.be/[A-Za-z0-9_-]+)', raw_url or '')
                                if not m:
                                    continue
                                clean_url = m.group(0)
                                video_id = m.group(0).split('v=')[-1].split('/')[-1]
                                title = v.get('title', 'Diagnostic Video')
                                # Remove tags from title
                                title = _re.sub(r'<[^>]+>', '', title).strip()
                                thumb = v.get('thumbnail') or f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg'
                                if 'default/default.jpg' in thumb:
                                    thumb = f'https://img.youtube.com/vi/{video_id}/mqdefault.jpg'
                                cleaned_vids.append({
                                    'url': clean_url,
                                    'title': title,
                                    'thumbnail': thumb,
                                    'video_id': video_id
                                })
                            text_output['youtube_videos'] = cleaned_vids
                            # Ensure content has no inline full video blocks
                            if 'content' in text_output and isinstance(text_output['content'], str):
                                c = text_output['content']
                                c = _re.sub(r'🎥 Watch Diagnostic Video', '', c)
                                # Remove any raw youtube watch URLs if structured vids exist
                                if cleaned_vids:
                                    c = _re.sub(r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+', '', c)
                                    c = _re.sub(r'https?://youtu\.be/[A-Za-z0-9_-]+', '', c)
                                text_output['content'] = c.strip()
                        
                        # Store text payload in agent for fallback retrieval
                        text_payload_json = json.dumps(text_output)
                        self._agent._diagnostic_text_payload = text_payload_json
                        self._text_payload = text_payload_json
                        self._has_text_payload = True
                        logger.info(f"📡 Prepared structured response: voice_len={len(voice_output)}, text_sources={len(text_output.get('web_sources', []))}, youtube_videos={len(text_output.get('youtube_videos', []))}")
                        
                        # Send diagnostic text via WebSocket to frontend (primary method)
                        asyncio.create_task(self._llm.send_diagnostic_text_via_websocket(text_payload_json))
                        
                        # Return ONLY voice for TTS (first chunk)
                        return voice_output
                
                # Check for direct structured format (legacy)
                elif (isinstance(parsed_response, dict) and 
                      'voice_output' in parsed_response and 'text_output' in parsed_response):
                    # This is the legacy structured response format
                    logger.info("📱 Received structured response from diagnostic agent (legacy format)")
                    
                    voice_output = parsed_response.get('voice_output', '')
                    text_output = parsed_response.get('text_output', {})
                    
                    # Store text payload in agent and locally for fallback
                    text_payload_json = json.dumps(text_output)
                    self._agent._diagnostic_text_payload = text_payload_json
                    self._text_payload = text_payload_json
                    self._has_text_payload = True
                    logger.info(f"📡 Prepared legacy structured response: voice_len={len(voice_output)}, text_payload_len={len(text_payload_json)}")
                    
                    # Send diagnostic text via WebSocket to frontend
                    asyncio.create_task(self._llm.send_diagnostic_text_via_websocket(text_payload_json))
                    
                    # Return ONLY voice for TTS
                    return voice_output
                else:
                    # Not a recognized structured response, return as-is (will be treated as voice only)
                    logger.info("📝 Received plain text response from diagnostic agent")
                    self._has_text_payload = False
                    return response
                    
            except (json.JSONDecodeError, TypeError):
                # Not JSON, return as-is
                logger.info("📝 Received non-JSON response from diagnostic agent")
                self._has_text_payload = False
                return response
                
        except Exception as e:
            logger.exception("Error getting response from diagnostic agent")
            self._has_text_payload = False
            return f"I apologize, but I encountered an error processing your request: {e}"

    async def aclose(self):
        """Clean up the stream."""
        if self._response_future and not self._response_future.done():
            self._response_future.cancel()
            try:
                await self._response_future
            except asyncio.CancelledError:
                pass
