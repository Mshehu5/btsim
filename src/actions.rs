use std::{iter::Sum, ops::Add};

use bitcoin::Amount;

use crate::{
    message::PayjoinProposal,
    wallet::{PaymentObligationData, PaymentObligationId, WalletHandleMut},
    TimeStep,
};

/// An Action a wallet can perform
pub(crate) enum Action {
    UnilateralSpend(PaymentObligationId),
    InitiatePayjoin(PaymentObligationId),
    ParticipateInPayjoin(PayjoinProposal),
    Wait,
}

/// Hypothetical outcomes of an action
pub(crate) enum Event {
    PaymentObligationHandled(PaymentObligationHandledEvent),
}

pub(crate) struct PaymentObligationHandledEvent {
    /// Payment obligation amount
    amount_handled: Amount,
    /// Balance difference after the action
    balance_difference: Amount,
    /// Time left on the payment obligation
    time_left: i32,
}

impl PaymentObligationHandledEvent {
    fn score(&self, payment_obligation_utility_factor: f64) -> ActionScore {
        // TODO: Utility should be higher for earlier deadlines.
        ActionScore(
            self.balance_difference
                .to_float_in(bitcoin::Denomination::Satoshi)
                - (payment_obligation_utility_factor
                    * self
                        .amount_handled
                        .to_float_in(bitcoin::Denomination::Satoshi)),
        )
    }
}

// TODO: implement EventCost for each event, each trait impl should define its own lambda weights

// Each strategy will prioritize some specific actions over other to minimize its wallet cost function
// E.g the unilateral spender associates high cost with batched transaction perhaps bc they dont like interactivity and don't care much for privacy
// They want to ensure they never miss a deadline. In that case the weights behind their deadline misses are high and batched payments will be low. i.e high payment anxiety
// TODO: should strategies do more than one thing per timestep?

// Cost function should evalute over unhandled payment obligations and payjoin / cospend oppurtunities. i.e Given all the payment obligations

/// State of the wallet that can be used to potential enumerate actions
#[derive(Debug, Default)]
pub(crate) struct WalletView {
    payment_obligations: Vec<PaymentObligationData>,
    current_timestep: TimeStep,
    // TODO: utxos, feerate, cospend oppurtunities, etc.
}

impl WalletView {
    pub(crate) fn new(
        payment_obligations: Vec<PaymentObligationData>,
        current_timestep: TimeStep,
    ) -> Self {
        Self {
            payment_obligations,
            current_timestep,
        }
    }
}

fn simulate_one_action(wallet_handle: &WalletHandleMut, action: &Action) -> Vec<Event> {
    let wallet_view = wallet_handle.wallet_view();
    let mut events = vec![];
    let old_info = wallet_handle.info().clone();
    let old_balance = wallet_handle.handle().effective_balance();

    // Deep clone the simulation
    let mut sim = wallet_handle.sim.clone();
    let mut predicated_wallet_handle = wallet_handle.data().id.with_mut(&mut sim);
    predicated_wallet_handle.do_action(action);
    let new_info = wallet_handle.info().clone();
    let new_balance = wallet_handle.handle().effective_balance();

    // Check for handled payment obligations -- we only handle one payment obligatoin per action. This may change in the future.
    // We may also want to evaluate bundles of actions.
    let handled_payment_obligations_diff = old_info
        .handled_payment_obligations
        .difference(new_info.handled_payment_obligations.clone())
        .into_iter()
        .next();
    if let Some(payment_obligation) = handled_payment_obligations_diff {
        let payment_obligation = payment_obligation.with(&sim).data();
        let deadline = payment_obligation.deadline;

        events.push(Event::PaymentObligationHandled(
            PaymentObligationHandledEvent {
                amount_handled: payment_obligation.amount,
                balance_difference: old_balance - new_balance,
                time_left: deadline.0 as i32 - wallet_view.current_timestep.0 as i32,
            },
        ));
    }

    events
}

/// Model payment obligation deadline anxiety as a cubic function of the time left.
/// The goal is to make the wallets more anxious as the deadline approaches and expires.
fn deadline_anxiety(deadline: i32, current_time: i32) -> f64 {
    // delta^3 / 50
    let time_left = deadline - current_time;
    (time_left.pow(3) as f64) / 50.0
}

/// Strategies will pick one action to minimize their cost
/// TODO: Strategies should be composible. They should enform the action decision space scoring and doing actions should be handling by something else that has composed multiple strategies.
pub(crate) trait Strategy {
    fn enumerate_candidate_actions(&self, state: &WalletView) -> impl Iterator<Item = Action>;
    fn do_something(&self, state: &WalletHandleMut) -> Action;
    fn score_action(&self, action: &Action, wallet_handle: &WalletHandleMut) -> ActionScore;
}

#[derive(Debug, PartialEq, PartialOrd)]
pub(crate) struct ActionScore(f64);

impl Sum for ActionScore {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.map(|s| s.0).sum())
    }
}

impl Eq for ActionScore {}

impl Ord for ActionScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        assert!(!self.0.is_nan() && !other.0.is_nan());
        self.0.partial_cmp(&other.0).expect("Checked for NaNs")
    }
}

impl Add for ActionScore {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

pub(crate) struct UnilateralSpender {
    pub(crate) payment_obligation_utility_factor: f64,
}

impl Strategy for UnilateralSpender {
    /// The decision space of the unilateral spender is the set of all payment obligations and payjoin proposals
    fn enumerate_candidate_actions(&self, state: &WalletView) -> impl Iterator<Item = Action> {
        if state.payment_obligations.is_empty() {
            return vec![Action::Wait].into_iter();
        }
        let mut actions = vec![];
        for po in state.payment_obligations.iter() {
            // For every payment obligation, we can spend it unilaterally
            actions.push(Action::UnilateralSpend(po.id));
        }

        actions.into_iter()
    }

    fn do_something(&self, state: &WalletHandleMut) -> Action {
        let wallet_view = state.wallet_view();
        // Unilateral spender ignores any payjoin or cospend oppurtunities
        self.enumerate_candidate_actions(&wallet_view)
            .min_by_key(|action| self.score_action(action, state))
            .unwrap_or(Action::Wait)
    }

    fn score_action(&self, action: &Action, wallet_handle: &WalletHandleMut) -> ActionScore {
        let events = simulate_one_action(wallet_handle, action);
        let mut score = ActionScore(0.0);
        for event in events {
            match event {
                Event::PaymentObligationHandled(event) => {
                    score = score + event.score(self.payment_obligation_utility_factor);
                }
                _ => (),
            };
        }
        score
    }
}
