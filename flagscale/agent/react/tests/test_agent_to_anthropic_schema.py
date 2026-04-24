"""Unit tests for _PluginShellTool.to_anthropic_schema method."""

import pytest

from flagscale.agent.react.agent import _PluginShellTool


class TestPluginShellToolToAnthropicSchema:
    """Test suite for _PluginShellTool.to_anthropic_schema method."""

    def test_to_anthropic_schema_normal_case(self):
        """Test to_anthropic_schema with normal input values."""
        spec = {
            "name": "test_tool",
            "description": "A test tool for testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Input parameter"}
                },
                "required": ["input"]
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert isinstance(result, dict)
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool for testing"
        assert result["input_schema"] == spec["parameters"]

    def test_to_anthropic_schema_empty_description(self):
        """Test to_anthropic_schema when description is empty (default value)."""
        spec = {
            "name": "empty_desc_tool",
            "parameters": {"type": "object", "properties": {}}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "empty_desc_tool"
        assert result["description"] == ""
        assert result["input_schema"] == {"type": "object", "properties": {}}

    def test_to_anthropic_schema_empty_parameters(self):
        """Test to_anthropic_schema when parameters is empty (default value)."""
        spec = {
            "name": "no_params_tool",
            "description": "Tool with no parameters"
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "no_params_tool"
        assert result["description"] == "Tool with no parameters"
        assert result["input_schema"] == {"type": "object", "properties": {}}

    def test_to_anthropic_schema_minimal_spec(self):
        """Test to_anthropic_schema with only required 'name' field."""
        spec = {"name": "minimal_tool"}
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "minimal_tool"
        assert result["description"] == ""
        assert result["input_schema"] == {"type": "object", "properties": {}}

    def test_to_anthropic_schema_complex_parameters(self):
        """Test to_anthropic_schema with complex parameter schema."""
        spec = {
            "name": "complex_tool",
            "description": "Tool with complex parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "string_param": {"type": "string", "minLength": 1, "maxLength": 100},
                    "number_param": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "bool_param": {"type": "boolean"},
                    "array_param": {"type": "array", "items": {"type": "string"}},
                    "object_param": {
                        "type": "object",
                        "properties": {
                            "nested": {"type": "string"}
                        }
                    },
                    "enum_param": {"enum": ["option1", "option2", "option3"]}
                },
                "required": ["string_param", "number_param"],
                "additionalProperties": False
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "complex_tool"
        assert result["description"] == "Tool with complex parameters"
        assert result["input_schema"] == spec["parameters"]

    def test_to_anthropic_schema_special_characters_in_name(self):
        """Test to_anthropic_schema with special characters in name."""
        spec = {
            "name": "tool-with_special.chars_123",
            "description": "Tool with special characters",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "tool-with_special.chars_123"
        assert result["description"] == "Tool with special characters"
        assert result["input_schema"] == {"type": "object"}

    def test_to_anthropic_schema_multiline_description(self):
        """Test to_anthropic_schema with multiline description."""
        spec = {
            "name": "multiline_tool",
            "description": """This is a multiline description.
It spans multiple lines.
And provides detailed information.""",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "multiline_tool"
        assert "\n" in result["description"]
        assert result["description"] == spec["description"]
        assert result["input_schema"] == {"type": "object"}

    def test_to_anthropic_schema_parameters_with_nested_objects(self):
        """Test to_anthropic_schema with deeply nested parameter objects."""
        spec = {
            "name": "nested_tool",
            "description": "Tool with nested parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "level1": {
                        "type": "object",
                        "properties": {
                            "level2": {
                                "type": "object",
                                "properties": {
                                    "level3": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "nested_tool"
        assert result["description"] == "Tool with nested parameters"
        assert result["input_schema"]["properties"]["level1"]["properties"]["level2"]["properties"]["level3"]["type"] == "string"

    def test_to_anthropic_schema_unicode_description(self):
        """Test to_anthropic_schema with unicode characters in description."""
        spec = {
            "name": "unicode_tool",
            "description": "测试工具 Test 工具 テストツール 🚀",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "unicode_tool"
        assert result["description"] == "测试工具 Test 工具 テストツール 🚀"
        assert result["input_schema"] == {"type": "object"}

    def test_to_anthropic_schema_long_name(self):
        """Test to_anthropic_schema with very long name."""
        long_name = "a" * 1000
        spec = {
            "name": long_name,
            "description": "Tool with very long name",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == long_name
        assert len(result["name"]) == 1000

    def test_to_anthropic_schema_long_description(self):
        """Test to_anthropic_schema with very long description."""
        long_description = "x" * 10000
        spec = {
            "name": "long_desc_tool",
            "description": long_description,
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["description"] == long_description
        assert len(result["description"]) == 10000

    def test_to_anthropic_schema_parameters_none_value(self):
        """Test to_anthropic_schema when parameters contains None values."""
        spec = {
            "name": "none_params_tool",
            "description": "Tool with None values",
            "parameters": {
                "type": "object",
                "properties": {
                    "optional_param": None,
                    "normal_param": {"type": "string"}
                }
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["name"] == "none_params_tool"
        assert result["input_schema"]["properties"]["optional_param"] is None
        assert result["input_schema"]["properties"]["normal_param"]["type"] == "string"

    def test_to_anthropic_schema_returns_new_dict_each_time(self):
        """Test that to_anthropic_schema returns a new dictionary object each time."""
        spec = {
            "name": "new_dict_tool",
            "description": "Test for new dict",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)

        result1 = tool.to_anthropic_schema()
        result2 = tool.to_anthropic_schema()

        assert result1 is not result2
        assert result1 == result2

    def test_to_anthropic_schema_dict_keys_are_correct(self):
        """Test that the returned dict has exactly the expected keys."""
        spec = {
            "name": "keys_test_tool",
            "description": "Test dict keys",
            "parameters": {"type": "object"}
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert set(result.keys()) == {"name", "description", "input_schema"}

    def test_to_anthropic_schema_all_fields_set(self):
        """Test that all fields in returned dict are properly set."""
        spec = {
            "name": "all_fields_tool",
            "description": "Test all fields",
            "parameters": {"type": "object", "properties": {"test": {"type": "string"}}}
        }
        tool = _PluginShellTool(spec)

        # Modify tool attributes after creation
        tool.name = "modified_tool"
        tool.description = "modified description"
        tool.parameters = {"type": "object"}

        result = tool.to_anthropic_schema()

        assert result["name"] == "modified_tool"
        assert result["description"] == "modified description"
        assert result["input_schema"] == {"type": "object"}

    def test_to_anthropic_schema_with_ref_schema(self):
        """Test to_anthropic_schema with JSON Schema $ref."""
        spec = {
            "name": "ref_tool",
            "description": "Tool with $ref",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"$ref": "#/definitions/Data"}
                },
                "definitions": {
                    "Data": {"type": "string"}
                }
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert result["input_schema"]["properties"]["data"]["$ref"] == "#/definitions/Data"
        assert result["input_schema"]["definitions"]["Data"]["type"] == "string"

    def test_to_anthropic_schema_with_any_of(self):
        """Test to_anthropic_schema with anyOf combinator."""
        spec = {
            "name": "anyof_tool",
            "description": "Tool with anyOf",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"}
                        ]
                    }
                }
            }
        }
        tool = _PluginShellTool(spec)
        result = tool.to_anthropic_schema()

        assert len(result["input_schema"]["properties"]["param"]["anyOf"]) == 2
        assert result["input_schema"]["properties"]["param"]["anyOf"][0]["type"] == "string"
        assert result["input_schema"]["properties"]["param"]["anyOf"][1]["type"] == "number"
