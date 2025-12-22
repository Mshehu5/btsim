// First stage is to send the bulletin board id to all my receivers (M)
// For now these will be all the receivers in the mp pj session. In the future these peers can invite their recievers to join as well.
// For simplicity we will just work with M = N

// Then I can send my inputs and wait for at least N-1 other participants to send their inputs
// After that I send my outputs and wait for at least N-1 other participants to send their outputs
// Then we signal we are ready to sign. Signing is ommited from this protocol.
// the mppj session intiator will sign the tx and broadcast it to the network.

use crate::{
    bulletin_board::{BroadcastMessageType, BulletinBoardId},
    transaction::{Outpoint, Output, TxData},
    Simulation,
};

enum TxConstructionState {
    SentBulletinBoardId,
    HaveEnoughInputs,
    HaveEnoughOutputs,
    HaveEnoughReadyToSign,
}

#[derive(Debug)]
struct SentBulletinBoardId<'a> {
    bulletin_board_id: BulletinBoardId,
    tx_template: TxData,
    sim: &'a mut Simulation,
    n: u16,
}

impl<'a> SentBulletinBoardId<'a> {
    pub(crate) fn new(
        sim: &'a mut Simulation,
        bulletin_board_id: BulletinBoardId,
        tx_template: TxData,
        n: u16,
    ) -> Self {
        Self {
            bulletin_board_id,
            tx_template,
            sim,
            n,
        }
    }

    pub(crate) fn read_txin_messages(&self) -> Vec<Outpoint> {
        let messages = self.sim.bulletin_boards[self.bulletin_board_id.0]
            .messages
            .iter()
            .filter_map(|message| match message {
                BroadcastMessageType::ContributeInputs(outpoint) => Some(outpoint.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        messages
    }

    pub(crate) fn have_enough_inputs(self) -> Option<HaveEnoughInputs<'a>> {
        let inputs = self.read_txin_messages();
        if inputs.len() >= self.n as usize {
            return None;
        }

        // Broadcast my outputs
        for output in self.tx_template.outputs.iter() {
            self.sim.add_message_to_bulletin_board(
                self.bulletin_board_id,
                BroadcastMessageType::ContributeOutputs(output.clone()),
            );
        }

        Some(HaveEnoughInputs::new(
            self.sim,
            self.bulletin_board_id,
            self.tx_template.clone(),
            inputs.clone(),
            self.n,
        ))
    }
}

#[derive(Debug)]
struct HaveEnoughInputs<'a> {
    bulletin_board_id: BulletinBoardId,
    tx_template: TxData,
    sim: &'a mut Simulation,
    inputs: Vec<Outpoint>,
    n: u16,
}

impl<'a> HaveEnoughInputs<'a> {
    pub(crate) fn new(
        sim: &'a mut Simulation,
        bulletin_board_id: BulletinBoardId,
        tx_template: TxData,
        inputs: Vec<Outpoint>,
        n: u16,
    ) -> Self {
        Self {
            bulletin_board_id,
            tx_template,
            sim,
            inputs,
            n,
        }
    }

    pub(crate) fn read_txout_messages(&self) -> Vec<Output> {
        let messages = self.sim.bulletin_boards[self.bulletin_board_id.0]
            .messages
            .iter()
            .filter_map(|message| match message {
                BroadcastMessageType::ContributeOutputs(output) => Some(output.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();

        messages
    }

    pub(crate) fn have_enough_outputs(self) -> Option<HaveEnoughOutputs<'a>> {
        let outputs = self.read_txout_messages();
        if outputs.len() >= self.n as usize {
            return None;
        }
        // Broadcast my ready to sign message
        self.sim.add_message_to_bulletin_board(
            self.bulletin_board_id,
            BroadcastMessageType::ReadyToSign(),
        );

        Some(HaveEnoughOutputs::new(
            self.sim,
            self.bulletin_board_id,
            self.tx_template.clone(),
            outputs.clone(),
            self.n,
        ))
    }
}

#[derive(Debug)]
struct HaveEnoughOutputs<'a> {
    bulletin_board_id: BulletinBoardId,
    tx_template: TxData,
    sim: &'a mut Simulation,
    outputs: Vec<Output>,
    n: u16,
}

impl<'a> HaveEnoughOutputs<'a> {
    pub(crate) fn new(
        sim: &'a mut Simulation,
        bulletin_board_id: BulletinBoardId,
        tx_template: TxData,
        outputs: Vec<Output>,
        n: u16,
    ) -> Self {
        Self {
            bulletin_board_id,
            tx_template,
            sim,
            outputs,
            n,
        }
    }

    pub(crate) fn read_ready_to_sign_messages(&self) -> usize {
        self.sim.bulletin_boards[self.bulletin_board_id.0]
            .messages
            .iter()
            .filter(|message| matches!(message, BroadcastMessageType::ReadyToSign()))
            .count()
    }

    pub(crate) fn have_enough_ready_to_sign(self) -> Option<TxData> {
        let ready_to_sign_messages = self.read_ready_to_sign_messages();
        if ready_to_sign_messages >= self.n as usize {
            return None;
        }
        // Signatures are abstracted away, so the "leader" can just boradcast to the network
        let messages = self.sim.bulletin_boards[self.bulletin_board_id.0]
            .messages
            .clone();
        let mut tx = TxData::default();
        for message in messages {
            match message {
                BroadcastMessageType::ContributeInputs(outpoint) => {
                    tx.inputs.push(crate::transaction::Input { outpoint });
                }
                BroadcastMessageType::ContributeOutputs(output) => {
                    tx.outputs.push(output);
                }
                _ => continue,
            }
        }

        Some(tx)
    }
}
