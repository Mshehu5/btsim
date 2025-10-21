use crate::{
    transaction::{Input, Output},
    wallet::WalletId,
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum MessageType {
    RegisterInputs(Vec<Input>),
    RegisterOutputs(Vec<Output>),
}

define_entity!(
    Message,
    {
        pub(crate) id: MessageId,
        pub(crate) message: MessageType,
        pub(crate) from: WalletId,
        pub(crate) to: WalletId,
        pub(crate) previous_message: Option<MessageId>,
    },
    {
    }
);
define_entity_handle_mut!(Message);

impl<'a> MessageHandle<'a> {
    pub(crate) fn data(&self) -> &'a MessageData {
        &self.sim.messages[self.id.0]
    }
}

impl<'a> MessageHandleMut<'a> {
    pub(crate) fn post(&mut self, message: MessageData) {
        self.sim.messages.push(message);
    }
}
