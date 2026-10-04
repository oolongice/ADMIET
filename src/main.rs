use admiet::{fit_many, FitOptions, Forest, Observations};
use std::{env, fs};

fn read_table(path: &str) -> Result<Vec<Vec<f64>>, String> {
    let contents = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let rows: Result<Vec<_>, _> = contents
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, line)| {
            line.split(|c: char| c.is_whitespace() || c == ',')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|e| format!("{path}: line {}: {e}", i + 1))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect();
    let rows = rows?;
    if rows.is_empty() {
        return Err(format!("{path}: empty file"));
    }
    Ok(rows)
}

fn read_dependencies(path: &str) -> Result<Vec<Vec<usize>>, String> {
    let contents = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line_number, line)| {
            line.split(|c: char| c.is_whitespace() || c == ',')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let index: usize = s.parse().map_err(|_| {
                        format!("{path}: line {}: invalid one-based index", line_number + 1)
                    })?;
                    index.checked_sub(1).ok_or_else(|| {
                        format!("{path}: line {}: indices start at 1", line_number + 1)
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn read_observations(time_path: &str, state_path: &str) -> Result<Observations, String> {
    let time_rows = read_table(time_path)?;
    let times = time_rows
        .iter()
        .map(|row| {
            if row.len() == 1 {
                Ok(row[0])
            } else {
                Err(format!("{time_path}: time file must have one column"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let states = read_table(state_path)?;
    if states.len() != times.len() {
        return Err("time and state files have different row counts".into());
    }
    let data = Observations { times, states };
    data.validate(data.states[0].len())?;
    Ok(data)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 {
        return Err("usage: admiet DATA_T DATA_X [--trajectory DATA_T DATA_X]... [--branches N] [--epochs N] [--pretrain N] [--lr RATE] [--dt MAX_STEP] [--window OBSERVATIONS] [--dependencies FILE] [--bounds LO HI] [--penalty VALUE] [--seed N]".into());
    }
    let mut data = vec![read_observations(&args[0], &args[1])?];
    let dimension = data[0].states[0].len();
    let mut branches = 1;
    let mut seed = 1;
    let mut bounds = (-1.0, 1.0);
    let mut dependencies = None;
    let mut options = FitOptions::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--branches" | "--epochs" | "--pretrain" | "--lr" | "--dt" | "--window"
            | "--dependencies" | "--penalty" | "--seed" => {
                let value = args.get(i + 1).ok_or("missing option value")?;
                match args[i].as_str() {
                    "--branches" => branches = value.parse().map_err(|_| "invalid branches")?,
                    "--epochs" => options.epochs = value.parse().map_err(|_| "invalid epochs")?,
                    "--pretrain" => {
                        options.pretrain_epochs =
                            value.parse().map_err(|_| "invalid pretrain epochs")?
                    }
                    "--lr" => options.learning_rate = value.parse().map_err(|_| "invalid rate")?,
                    "--dt" => options.max_step = value.parse().map_err(|_| "invalid step")?,
                    "--window" => {
                        options.shooting_window = value.parse().map_err(|_| "invalid window")?
                    }
                    "--dependencies" => dependencies = Some(read_dependencies(value)?),
                    "--penalty" => {
                        options.one_hot_penalty = value.parse().map_err(|_| "invalid penalty")?
                    }
                    "--seed" => seed = value.parse().map_err(|_| "invalid seed")?,
                    _ => unreachable!(),
                }
                i += 2;
            }
            "--bounds" => {
                bounds.0 = args
                    .get(i + 1)
                    .ok_or("missing lower bound")?
                    .parse()
                    .map_err(|_| "invalid lower bound")?;
                bounds.1 = args
                    .get(i + 2)
                    .ok_or("missing upper bound")?
                    .parse()
                    .map_err(|_| "invalid upper bound")?;
                i += 3;
            }
            "--trajectory" => {
                let time_path = args.get(i + 1).ok_or("missing trajectory time file")?;
                let state_path = args.get(i + 2).ok_or("missing trajectory state file")?;
                let additional = read_observations(time_path, state_path)?;
                additional.validate(dimension)?;
                data.push(additional);
                i += 3;
            }
            _ => return Err(format!("unknown option {}", args[i])),
        }
    }
    let mut model = if let Some(inputs) = dependencies {
        Forest::with_dependencies(dimension, branches, inputs, bounds, seed)?
    } else {
        Forest::new(dimension, branches, bounds, seed)?
    };
    let report = fit_many(&mut model, &data, &options)?;
    println!(
        "initial loss: {:.8}\nfinal loss: {:.8}",
        report.initial_loss, report.final_loss
    );
    for (i, equation) in model.equations().iter().enumerate() {
        println!("dx{}/dt = {equation}", i + 1);
    }
    Ok(())
}
