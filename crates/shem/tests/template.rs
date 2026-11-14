//! Go-style templates, as `shem images --format` takes them.

use serde_json::json;
use shem::template::Template;

fn render(source: &str) -> String {
    let data = json!({
        "Repository": "hf.co/org/model",
        "Tag": "latest",
        "Bytes": 1234,
        "Labels": ["a", "b"],
        "Nested": { "Name": "inner" },
    });
    Template::parse(source).unwrap().render(&data).unwrap()
}

fn error(source: &str) -> String {
    let result = Template::parse(source).and_then(|t| t.render(&json!({ "Tag": "x" })));
    format!("{:#}", result.unwrap_err())
}

#[test]
fn fields_render_between_text() {
    assert_eq!(render("{{.Repository}}:{{.Tag}}"), "hf.co/org/model:latest");
    assert_eq!(render("{{ .Nested.Name }} {{.Bytes}}"), "inner 1234");
    assert_eq!(render("no actions"), "no actions");
    assert_eq!(render(r"{{.Tag}}\t{{.Bytes}}\n"), "latest\t1234\n");
}

#[test]
fn functions_take_arguments_and_piped_values_last() {
    assert_eq!(render("{{upper .Tag}}"), "LATEST");
    assert_eq!(render("{{.Tag | upper}}"), "LATEST");
    assert_eq!(render("{{.Tag | title | lower}}"), "latest");
    assert_eq!(render("{{title \"two words\"}}"), "Two Words");
    assert_eq!(render("{{truncate .Repository 5}}"), "hf.co");
    assert_eq!(render("{{pad .Tag 1 2}}|"), " latest  |");
    assert_eq!(render("{{join .Labels \",\"}}"), "a,b");
}

#[test]
fn values_print_as_json_unless_strings() {
    assert_eq!(render("{{json .Tag}}"), "\"latest\"");
    assert_eq!(render("{{.Labels}}"), "[\"a\",\"b\"]");
    assert_eq!(render("{{len .Labels}} {{len .Tag}}"), "2 6");
    assert_eq!(render("{{split .Repository \"/\" | len}}"), "3");
    assert_eq!(render("{{\"a\\\"b\"}} {{-1}}"), "a\"b -1");
}

#[test]
fn trim_markers_eat_the_space_beside_them() {
    assert_eq!(render("a  {{- .Tag -}}  b"), "alatestb");
    assert_eq!(render("{{.Tag}} {{-3}}"), "latest -3");
}

#[test]
fn mistakes_are_named() {
    assert!(error("{{.Nope}}").contains("no field .Nope: there are Tag"));
    assert!(error("{{frob .Tag}}").contains("no function \"frob\""));
    assert!(error("{{.Tag").contains("unclosed action"));
    assert!(error("{{}}").contains("empty action"));
    assert!(error("{{truncate .Tag}}").contains("truncate takes 2"));
    assert!(error("{{.Tag | .Tag}}").contains("only a function"));
    assert!(error("{{upper \"x}}").contains("unclosed string"));
    assert!(error("{{truncate .Tag \"many\"}}").contains("expected a count"));
}
