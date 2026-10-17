// A setup form card. A field explains what's wrong once the person has typed in it and moved
// on. Submit is never disabled (people can't find out why a greyed-out button won't work):
// submitting with problems shows the GOV.UK-style error summary at the top, focused, with a
// link to each field, and "Error: " at the start of the page title. Submit shows a spinner
// while saving. Inputs are uncontrolled and validated with the browser's constraint API.
import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";

export interface FieldSpec {
  name: string;
  label: string;
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  placeholder?: string;
  defaultValue?: string;
  autoComplete?: string;
  spellCheck?: boolean;
}

interface Props {
  title: string;
  hint?: string;
  fields: FieldSpec[];
  submitLabel: string;
  /** Clear the fields after a successful submit (for forms that add one thing after another). */
  resetOnSuccess?: boolean;
  /** Returns an optional success message. */
  onSubmit(values: Record<string, string>): Promise<string | void>;
}

function message(input: HTMLInputElement, label: string): string {
  const v = input.validity;
  if (v.valid) return "";
  if (v.valueMissing) return `Enter the ${/^[A-Z]{2}/.test(label.split(" ").pop() ?? "") ? label : label.toLowerCase()}.`;
  if (v.tooShort) return `${label} needs at least ${input.minLength} characters.`;
  return input.validationMessage;
}

export function FormCard({ title, hint, fields, submitLabel, resetOnSuccess, onSubmit }: Props) {
  const formRef = useRef<HTMLFormElement>(null);
  const id = useId();
  const summaryRef = useRef<HTMLDivElement>(null);
  const [summary, setSummary] = useState<{ inputId: string; text: string }[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; error: boolean }>({ text: "", error: false });

  const validate = (input: HTMLInputElement) => {
    const label = fields.find((f) => f.name === input.name)?.label ?? "This field";
    setErrors((e) => ({ ...e, [input.name]: message(input, label) }));
  };

  const submit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    if (!form.checkValidity()) {
      const problems: { inputId: string; text: string }[] = [];
      for (const input of form.querySelectorAll("input")) {
        validate(input);
        const label = fields.find((f) => f.name === input.name)?.label ?? "This field";
        if (!input.validity.valid) problems.push({ inputId: input.id, text: message(input, label) });
      }
      setTouched(Object.fromEntries(fields.map((f) => [f.name, true])));
      setSummary(problems);
      requestAnimationFrame(() => summaryRef.current?.focus());
      return;
    }
    setSummary([]);
    setMsg({ text: "", error: false });
    setBusy(true);
    try {
      const values = Object.fromEntries(new FormData(form).entries()) as Record<string, string>;
      setMsg({ text: (await onSubmit(values)) || "", error: false });
      if (resetOnSuccess) {
        form.reset();
        setTouched({});
      }
    } catch (err) {
      setMsg({ text: err instanceof ApiError ? err.detail : (err as Error).message, error: true });
    } finally {
      setBusy(false);
    }
  };

  // "Error: " in the page title while the summary shows, so screen readers hear it first.
  useEffect(() => {
    if (!summary.length) return;
    const title = document.title.replace(/^Error: /, "");
    document.title = `Error: ${title}`;
    return () => {
      document.title = title;
    };
  }, [summary.length]);

  return (
    <form
      ref={formRef}
      className="card setup-card"
      noValidate
      onSubmit={submit}
      onInput={(e) => {
        const input = e.target;
        if (!(input instanceof HTMLInputElement)) return;
        setTouched((t) => ({ ...t, [input.name]: true }));
        if (errors[input.name]) validate(input); // clear as soon as it's fixed
        if (summary.length && e.currentTarget.checkValidity()) setSummary([]);
      }}
      onBlur={(e) => {
        const input = e.target;
        if (input instanceof HTMLInputElement && touched[input.name]) validate(input);
      }}
    >
      <h2>{title}</h2>
      {hint && <p className="hint">{hint}</p>}
      {summary.length > 0 && (
        <div ref={summaryRef} className="error-summary" role="alert" tabIndex={-1} aria-labelledby={`${id}-problem`}>
          <h3 id={`${id}-problem`}>There is a problem</h3>
          <ul>
            {summary.map((p) => (
              <li key={p.inputId}>
                <a
                  href={`#${p.inputId}`}
                  onClick={(e) => {
                    e.preventDefault(); // the app routes on the hash; just move to the field
                    document.getElementById(p.inputId)?.focus();
                  }}
                >
                  {p.text}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
      {fields.map((f) => {
        const inputId = `${id}-${f.name}`;
        const error = errors[f.name] ?? "";
        return (
          <div className="field" key={f.name}>
            <label htmlFor={inputId}>
              {f.label}
              {!f.required && <span className="optional"> (optional)</span>}
            </label>
            <input
              id={inputId}
              name={f.name}
              required={f.required}
              minLength={f.minLength}
              maxLength={f.maxLength}
              placeholder={f.placeholder}
              defaultValue={f.defaultValue}
              autoComplete={f.autoComplete}
              spellCheck={f.spellCheck}
              aria-invalid={error ? true : undefined}
              aria-describedby={`${inputId}-error`}
            />
            <span className="field-error" id={`${inputId}-error`} aria-live="polite">
              {error}
            </span>
          </div>
        );
      })}
      <button type="submit" className="primary" disabled={busy} aria-busy={busy}>
        {submitLabel}
      </button>
      <p className={msg.error ? "form-msg error" : "form-msg"} role="status">
        {msg.text}
      </p>
    </form>
  );
}
