import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import anthropic
import os
from retry import retry


try:
    anthr = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    anthr_models = [
        "claude-2.1",
        "claude-3-opus-20240229"
    ]
except:
    anthr = None
    anthr_models = []
    
    
def dict_to_xml(tag, d):
    """
    Recursively turn a dict of key/value pairs into XML,
    including handling nested dicts.
    """
    elem = ET.Element(tag)
    for key, val in d.items():
        if isinstance(val, dict):
            # If the value is a dict, make a recursive call
            child = dict_to_xml(key, val)
        else:
            child = ET.Element(key)
            child.text = str(val)
        elem.append(child)
    return elem

def dict_to_xml_str(tag, d):
    elem = dict_to_xml(tag, d)
    rough_string = ET.tostring(elem, encoding='unicode')
    # Pretty-print the XML
    reparsed = parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def list_of_dicts_to_xml_str(list_of_dicts, root_tag="tools", child_tag="tool_definition", indent=True):
    """
    Convert a list of dictionaries to an XML string, with optional pretty-printing.
    """
    root = ET.Element(root_tag)
    for d in list_of_dicts:
        root.append(dict_to_xml(child_tag, d))
    
    # Convert the ElementTree to a string
    rough_string = ET.tostring(root, encoding='unicode')
    if indent:
        # Pretty-print the XML
        reparsed = parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    else:
        return rough_string



def xml_to_dict(elem):
    d = {}
    for child in elem:
        if child.tag not in d:
            d[child.tag] = []
        d[child.tag].append(xml_to_dict(child) if child else child.text)
    for k in d:
        d[k] = d[k][0] if len(d[k]) == 1 else d[k]
        if isinstance(d[k], dict) and not d[k]:  # Removing empty dictionaries
            d[k] = None
    return d




def example():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to one or many of the test subjects",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "The reasoning behind the message"
                        },
                        "message": {
                            "type": "string",
                            "description": "The message to send"
                        },
                        "recipients": {
                            "type": "string",
                            "description": "Commaseparated list of nicknames to send the message to"
                        }
                    },
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "guess",
                "description": "Guess who the target is",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "The reasoning behind the message"
                        },
                        "guess": {
                            "type": "string",
                            "description": "The nickname of the guess"
                        }
                    },
                }
            }
        }
    ]

    xml_string = list_of_dicts_to_xml_str(tools)
    print(xml_string)


    action = {
        "name": "send_message",
        "parameters": {
            "reasoning": "This is an example",
            "message": "Hello",
            "recipients": 'Alan, Max'
        }
    }
    action_str = dict_to_xml_str("action", action)
    print(action_str)


example_action = {
    "name": "example_action",
    "parameters": {
        "info": "Your response should follow the schema of the described actions, this is just an example for the xml syntax",
        "some_number": 42,
    }
}


@retry(tries=3)
def get_response_anthropic_with_tools(
    model,
    history,
    temperature,
    *,
    max_tokens=1024,
    tools=[],
    example_action=example_action,
    **kwargs
):
    system_msg = [msg for msg in history if msg['role']=='system']
    system_msg = '' if len(system_msg) == 0 else system_msg[0]['content']
    
    system_msg += f'\n\nUse one of the available tools to choose an action.\n{list_of_dicts_to_xml_str(tools)}\n\n'
    system_msg += f'Respond in valid XML, for example:\n\n{dict_to_xml_str("action", example_action)}\Pick one action at a time.'
    
    history = [msg for msg in history if not msg['role']=='system']
    message = anthr.messages.create(
        model=model,
        messages=history,
        temperature=temperature,
        max_tokens=max_tokens,
        system=system_msg,
        **kwargs
    )
    response_msg = {
        'role': 'assistant',
        'content': message.content[0].text
    }
    
    with open("last_claude_response_raw.txt", "w") as f:
        f.write(message.content[0].text)
    
        
    xml_elem = ET.fromstring(message.content[0].text)
    parsed_response = xml_to_dict(xml_elem)
    action = {
            "name": parsed_response['name'],
            "args": parsed_response['parameters']
        }
    return response_msg, action
    
    

