import discord
from discord.ext import commands
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import asyncio
from typing import Optional
import logging
import traceback

from juglogger import log_interaction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# Model Configuration
# -----------------------------
class DiscordBotModel:
    def __init__(self, model_dir: str = "lora-chat-out"):       # Update to newest version of Jugo LoRA model
        self.model_dir = model_dir
        self.base_model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        
    def load_model(self):
        """Load the model and tokenizer"""
        if self.is_loading:
            return False
            
        self.is_loading = True
        try:
            logger.info("Loading model...")
            
            # Configure quantization
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            
            # Load tokenizer from fine-tuned directory
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with quantization
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_dir)
            self.model.eval()
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
            return False
        finally:
            self.is_loading = False
    
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response from the model"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please try again later."
        
        try:
            # Tokenize the prompt directly (it's already formatted)
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the bot's response (after the prompt)
            # The prompt is already in the response, so we need to remove it
            if full_response.startswith(prompt):
                response = full_response[len(prompt):].strip()
            else:
                # If format didn't match, try to find "Bot:" and extract after that
                if "Bot:" in full_response:
                    # Get everything after the last "Bot:"
                    parts = full_response.split("Bot:")
                    response = parts[-1].strip()
                else:
                    # Fallback: just return everything
                    response = full_response
            
            # Clean up the response
            response = response.replace("<s>", "").replace("</s>", "").strip()
            
            # Truncate if too long
            if len(response) > 3600:
                response = response[:3597] + "..."

            # Log the interaction

            # Extract the user's message from the prompt
            if "User:" in prompt:
                # Take everything after the last "User:"
                log_prompt = prompt.split("User:")[-1].strip()
                # If "Bot:" appears, remove it along with anything after it
                if "Bot:" in log_prompt:
                    log_prompt = log_prompt.split("Bot:")[0].strip()
            else:
                # Fallback: log the full prompt as user message
                log_prompt = prompt.strip()

            # Safety: ensure user message is not empty
            if not log_prompt:
                log_prompt = "<empty_user_message>"
            log_interaction(log_prompt, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return f"Error: {str(e)[:100]}"

# -----------------------------
# Discord Bot
# -----------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)
model_handler = DiscordBotModel()

# Track conversations per channel/user
conversation_history = {}
MAX_HISTORY = 12

def add_to_history(channel_id: int, user_id: int, message: str, is_bot: bool = False):
    """Maintain conversation history per channel/user"""
    key = f"{channel_id}_{user_id}"
    if key not in conversation_history:
        conversation_history[key] = []
    
    role = "Bot" if is_bot else "User"
    conversation_history[key].append(f"{role}: {message}")
    
    # Keep only last MAX_HISTORY messages
    if len(conversation_history[key]) > MAX_HISTORY:
        conversation_history[key] = conversation_history[key][-MAX_HISTORY:]

def get_conversation_context(channel_id: int, user_id: int, new_message: str) -> str:
    """Get recent conversation history"""
    key = f"{channel_id}_{user_id}"

    system_prompt = (
        "System: You are Jugo, a chaotic and funny AI Discord chatbot. "
        "You always stay in character, making witty and humorous remarks. Your goals are: "
        "1) Entertain the user, 2) Keep the conversation engaging, "
        "3) Never break character. "
    )

    if key not in conversation_history:
        return system_prompt + f"User: {new_message}\nBot:"
    
    # Get the conversation history
    history = conversation_history[key]
    
    # Add the new user message temporarily for context
    temp_history = history + [f"User: {new_message}"]
    
    # Build the conversation string
    """ conversation_string = "\n".join(temp_history)
    
    # Add "Bot:" at the end for the model to complete
    formatted_prompt = f"{conversation_string}\nBot:" """
    
    return system_prompt + "\n".join(temp_history) + "\nBot:"

@bot.event
async def on_ready():
    """Called when the bot is ready"""
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    
    # Load the model
    success = model_handler.load_model()
    if success:
        logger.info("Model loaded and ready for use!")
    else:
        logger.error("Failed to load model!")

@bot.event
async def on_message(message):
    """Handle incoming messages"""
    # Don't respond to ourselves
    if message.author == bot.user:
        return
    
    if model_handler.is_loading:
        await message.reply("⏳ Model is still loading, please wait 3-5 seconds...", mention_author=True)
        return

    if model_handler.model is None:
        await message.reply("❌ Model not loaded. Try again in a moment.", mention_author=True)
        return
    
    # Check if bot is mentioned OR message is a DM
    bot_mentioned = bot.user.mentioned_in(message) and not message.mention_everyone
    is_dm = isinstance(message.channel, discord.DMChannel)
    
    # Also respond to messages starting with !chat
    is_chat_command = message.content.startswith("!chat")
    
    if bot_mentioned or is_dm or is_chat_command:
        # Show typing indicator
        async with message.channel.typing():
            try:
                # Extract clean message content
                if bot_mentioned:
                    # Remove the mention from the message
                    clean_content = message.content.replace(f'<@{bot.user.id}>', '').strip()
                elif is_chat_command:
                    clean_content = message.content.replace('!chat', '', 1).strip()
                else:
                    clean_content = message.content
                
                if not clean_content:
                    clean_content = "Hello!"
                
                # Get conversation context with new message included
                formatted_prompt = get_conversation_context(
                    message.channel.id, 
                    message.author.id, 
                    clean_content
                )

                logger.info(f"Prompt length: {len(formatted_prompt)} characters")
                
                
                # Add user message to history
                add_to_history(message.channel.id, message.author.id, clean_content)
                
                # Generate response
                response = model_handler.generate_response(formatted_prompt, max_length=150)

                if response.startswith("Error:") or response.startswith("Model not loaded"):
                    # Send the error message and stop
                    await message.reply(response, mention_author=True)
                    return
                
                logger.info(f"Generated response: {response[:100]}...")
                
                # Add bot response to history
                add_to_history(message.channel.id, message.author.id, response, is_bot=True)
                
                # Send response
                await message.reply(response, mention_author=True)
                
            except Exception as e:
                logger.error(f"Error in on_message: {e}")
                logger.error(traceback.format_exc())
                # Only send ONE error message
                await message.reply("Sorry, I encountered an error processing your message.", mention_author=True)
    
    # Process commands as well
    await bot.process_commands(message)

@bot.command(name="clear")
async def clear_history(ctx):
    """Clear conversation history for this channel/user"""
    key = f"{ctx.channel.id}_{ctx.author.id}"
    if key in conversation_history:
        del conversation_history[key]
    await ctx.send("Conversation history cleared!", delete_after=5)

@bot.command(name="ping")
async def ping(ctx):
    """Check if the bot is responsive"""
    latency = round(bot.latency * 1000)
    await ctx.send(f"Pong! Latency: {latency}ms")

@bot.command(name="modelinfo")
async def model_info(ctx):
    """Show model information"""
    if model_handler.model is not None:
        await ctx.send("🤖 Model: Zephyr-7B with Discord fine-tuning\n✅ Status: Loaded and ready")
    else:
        await ctx.send("❌ Model is not loaded. Please check the logs.")

@bot.command(name="generate")
async def generate(ctx, *, prompt: str):
    """Generate text from a prompt"""
    async with ctx.channel.typing():
        formatted_prompt = f"User: {prompt}\nBot:"
        response = model_handler.generate_response(formatted_prompt, max_length=200)
        await ctx.send(f"**Response:** {response}")

@bot.command(name="reload")
@commands.has_permissions(administrator=True)
async def reload_model(ctx):
    """Reload the model (admin only)"""
    await ctx.send("🔄 Reloading model...")
    success = model_handler.load_model()
    if success:
        await ctx.send("✅ Model reloaded successfully!")
    else:
        await ctx.send("❌ Failed to reload model!")

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Try `!help` for available commands.")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("You don't have permission to use this command.")
    else:
        logger.error(f"Command error: {error}")
        await ctx.send(f"An error occurred: {str(error)}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load your Discord bot token from environment variable or file
    import os
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file

    """ print("Loading model before starting bot...")
    model_handler.load_model() """
    
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    
    if not DISCORD_TOKEN:
        print("Error: DISCORD_TOKEN not found in environment variables!")
        print("Create a .env file with: DISCORD_TOKEN=your_token_here")
        print("Or set the environment variable directly.")
        exit(1)
    
    bot.run(DISCORD_TOKEN)