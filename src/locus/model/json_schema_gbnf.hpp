#pragma once

#include <string>
#include <string_view>

namespace locus::model {

/**
 * A GBNF grammar (rule "root") for the JSON text that a JSON Schema
 * accepts (i#402), for constrained decoding. Properties come out in
 * the order the schema lists them.
 *
 * Supported: type (one or a list), const, enum, anyOf, oneOf, allOf
 * with one member, $ref to #/$defs/... or #/definitions/... (recursion
 * allowed), properties with required (optional ones may be left out),
 * additionalProperties (false when properties are given and it is not
 * set), items, prefixItems, minItems, maxItems, minLength, maxLength,
 * pattern (the whole string must match), and format date, time,
 * date-time and uuid. Numeric bounds and other keywords are ignored:
 * the output still parses, but may not validate.
 *
 * @param schema_json The schema, as JSON text. Its nesting must already
 *     be bounded (from http::parse_json): the conversion recurses once
 *     per level.
 * @returns The grammar.
 * @throws std::invalid_argument on a schema it cannot express.
 */
std::string json_schema_to_gbnf(std::string_view schema_json);

/**
 * @returns A GBNF grammar (rule "root") for any JSON value, the
 *     response_format json_object mode.
 */
std::string json_value_gbnf();

}  // namespace locus::model
